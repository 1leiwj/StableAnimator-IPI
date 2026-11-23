#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

driver=./animation_data/ceshi/10002
python - "$driver/poses_dense133.npy" <<'PY'
import sys
import numpy as np

try:
    pose = np.load(sys.argv[1], allow_pickle=False)
    if pose.ndim != 3 or pose.shape[1:] != (133, 2) or len(pose) == 0:
        raise ValueError(f"expected [frames,133,2], found {pose.shape}")
except (OSError, ValueError, EOFError) as error:
    raise SystemExit(f"Cannot read complete driving pose {sys.argv[1]}: {error}")
print(f"Driver 10002: {len(pose)} frames", flush=True)
PY

match_ankle="${MATCH_ANKLE_HEIGHT:-0}"
case "$match_ankle" in
    0) label=root0 ;;
    1) label=ankle_height
       if ! grep -Fq -- '--match_ankle_height' command_ipi_infer.sh; then
           echo "Install the ankle-height update before running MATCH_ANKLE_HEIGHT=1." >&2
           exit 1
       fi ;;
    *) echo "MATCH_ANKLE_HEIGHT must be 0 or 1." >&2; exit 2 ;;
esac

# Keep the 31000 root0 settings for the new driver; process the entire sequence.
env CHECKPOINT_ROOT=./checkpoints/Animation_ipi_7ganjing CHECKPOINT_STEP=31000 \
    IPI_TEMPORAL_V2=1 NO_IPI=1 RETARGET=1 POSE_ROOT_MOTION_SCALE=0 MATCH_ANKLE_HEIGHT="$match_ankle" \
    REFERENCE_IMAGE=./basic_infer/ces/30555.jpg DRIVING_DIR="$driver" POSE_COORDINATE_SPACE=normalized \
    FRAME_START=0 FRAME_END= SOURCE_TIMELINE=0 WIDTH=512 HEIGHT=768 \
    POSE_SCALE=3 IPI_SCALE=1 TILE_SIZE=20 FRAMES_OVERLAP=6 NUM_INFERENCE_STEPS=20 \
    GUIDANCE_SCALE=3 MIN_GUIDANCE_SCALE=2 MAX_GUIDANCE_SCALE=3.5 \
    REFERENCE_LATENT_SCALE=1 REFERENCE_CLIP_SCALE=1 INDEPENDENT_NOISE=0 SEED=23123134 \
    IS_STATUE=0 MOTION_BUCKET_ID=300 FPS_TIME_ID=6 NOISE_AUG_STRENGTH=0 \
    DECODE_CHUNK_SIZE=4 SKIP_GIF=1 OUTPUT_DIR="${OUTPUT_DIR:-./basic_infer/10002_to_30555_31000_${label}_full}" \
    bash command_ipi_infer.sh
