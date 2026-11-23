"""Replay a single-window IPI run up to one noise level and compare motion conditions."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(PROJECT))  # Keep the replay entry bundled in the .pyz first.
from inference_latent_ar import run_inference
from animation.modules.ipi_joint_motion import JointMotionCondition


def rms(x):
    return torch.linalg.vector_norm(x, dtype=torch.float32) / math.sqrt(x.numel())


def relative_rms(x, scale):
    divisor = scale.item()
    return (rms(x).item() / divisor) if divisor > 0 else None


def measure_layer(stack, block, result):
    # Capture the same projected vector and temporal activation used in the actual addition.
    def projected(module, args, hidden):
        result["_before_gate"] = rms(hidden)
        result["_residual"] = (block.ipi_temporal_gate.tanh() * hidden).detach()

    def temporal(module, args, hidden):
        r = result["_residual"].to(hidden.dtype).float()
        scale = rms(hidden)
        result["_temporal_scale"] = scale
        before = result.pop("_before_gate")
        result["before_gate_over_temporal"] = (before / scale).item() if scale.item() > 0 else None
        result["residual_over_temporal"] = relative_rms(r, scale)
        result["frame_delta_over_temporal"] = relative_rms(r.diff(dim=0), scale)

    stack.enter_context(block.ipi_temporal_projection.register_forward_hook(projected))
    stack.enter_context(block.temporal_transformer_blocks[-1].register_forward_hook(temporal))


@torch.no_grad()
def probe(pipeline, inputs):
    unet = pipeline.unet
    frames = inputs["num_frames"]
    step_index = (inputs["num_inference_steps"] - 1) // 2
    captured = {}

    class FoundConditional(Exception):
        pass

    def capture(module, args, kwargs):
        if args[1].item() != pipeline.scheduler.timesteps[step_index].item():
            return
        if not torch.count_nonzero(kwargs["pose_latents"]).item():
            return  # The normal CFG unconditional pass uses a zero PoseNet tensor.
        if list(kwargs["global_frame_indices"]) != list(range(frames)):
            raise ValueError("Expected the entire clip in one ordered window.")
        captured.update(args=args, kwargs=kwargs)
        raise FoundConditional

    def check_capture(pipe, i, t, state):
        if i >= step_index:
            raise RuntimeError("No nonzero conditional PoseNet input at the requested timestep.")
        return state

    with unet.register_forward_pre_hook(capture, with_kwargs=True):
        try:
            pipeline(**dict(inputs, output_type="latent", callback_on_step_end=check_capture))
        except FoundConditional:
            pass
    args, kwargs = captured["args"], captured["kwargs"]
    aligned = kwargs["precomputed_ipi_features"]
    localized = isinstance(aligned, JointMotionCondition)
    aligned_tokens = aligned.tokens if localized else aligned
    if aligned_tokens.ndim != 3 or aligned_tokens.shape[0] != frames or frames < 2:
        raise ValueError("Expected one clip with per-frame IPI tokens.")
    pose = inputs["dwpose_raw"]
    clip = inputs["clip_image_embeds"]
    frozen = unet.ipi_adapter(pose[:, :1].expand_as(pose), clip[:, :1].expand_as(clip), ipi_temporal_v2=True)
    frozen_tokens = frozen.tokens if localized else frozen

    def predict(tokens):
        return unet(*args, **dict(kwargs, precomputed_ipi_features=tokens))[0].float()

    def traced_predict(tokens):
        layers = {}
        with ExitStack() as stack:
            for name, block in (("mid", unet.mid_block.attentions[0]), ("last_up", unet.up_blocks[-1].attentions[-1])):
                layers[name] = {}
                measure_layer(stack, block, layers[name])
            return predict(tokens), layers

    prediction, layers = traced_predict(aligned)
    static_prediction, static_layers = traced_predict(frozen)
    for name, layer in layers.items():
        delta = layer.pop("_residual").float() - static_layers[name]["_residual"].float()
        layer["aligned_vs_frozen_residual_over_temporal"] = relative_rms(delta, layer.pop("_temporal_scale"))
    scale = rms(prediction)
    conditions = {
        "aligned_repeat": relative_rms(prediction - predict(aligned), scale),
        "aligned_vs_first_frame_repeated": relative_rms(prediction - static_prediction, scale),
        "aligned_vs_off": relative_rms(prediction - predict(None), scale),
    }
    if localized:
        conditions["aligned_vs_joint_disabled"] = relative_rms(prediction - predict(aligned_tokens), scale)
        # Hold the original 16 tokens fixed and change only the new branch's driver.
        frozen_joints = JointMotionCondition(aligned_tokens, frozen.centers, frozen.visible, frozen.values)
        conditions["aligned_vs_only_joints_frozen"] = relative_rms(prediction - predict(frozen_joints), scale)
    return {
        "denoising_step_1_based": step_index + 1,
        "denoising_steps_total": inputs["num_inference_steps"],
        "timestep": args[1].item(),
        "aligned_layers": layers,
        "tokens_aligned_vs_frozen_relative_rms": relative_rms(aligned_tokens.float() - frozen_tokens.float(), rms(aligned_tokens)),
        "pooled_aligned_vs_frozen_relative_rms": relative_rms(
            aligned_tokens.mean(dim=1).float() - frozen_tokens.mean(dim=1).float(), rms(aligned_tokens.mean(dim=1))),
        "conditional_prediction_relative_rms": conditions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        default=PROJECT / "basic_infer/10002_ipi_limb_pilot_step1000/raise/on")
    run_dir = parser.parse_args().run_dir.resolve()
    os.chdir(PROJECT)  # The paths recorded by the original launcher are project-relative.
    settings = json.loads((run_dir / "inference_settings.json").read_text(encoding="utf-8"))
    if (settings["ipi_mode"] != "v2" or settings["no_ipi"] or not settings["retarget_pose"]
            or not 2 <= settings["num_frames"] <= settings["tile_size"]):
        parser.error("Supply a retargeted, single-window IPI v2 ON result with at least two frames.")
    with tempfile.TemporaryDirectory(prefix="stableanimator-ipi-probe-") as scratch:
        replay = argparse.Namespace(**dict(settings, output_dir=scratch))

        def inspect(pipeline, inputs):
            filename = "retargeted_poses_dense133.npy"
            if not np.array_equal(np.load(run_dir / filename), np.load(Path(scratch) / filename)):
                raise ValueError("The replayed pose differs from the saved run's target.")
            return probe(pipeline, inputs)

        report = run_inference(replay, pipeline_probe=inspect)
    report.update(source_frames=[settings["frame_start"], settings["frame_end"] - 1],
                  seed=settings["seed"], unet=settings["unet_model_name_or_path"])
    contents = json.dumps(report, indent=2, allow_nan=False)
    output = run_dir / "ipi_motion_probe.json"
    output.write_text(contents + "\n", encoding="utf-8")
    print(contents, flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
