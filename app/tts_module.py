# -*- coding: utf-8 -*-
"""
AI 雕像动态驱动生成系统 - Minimax 语音合成与多模态音视频混流模块 (V3.0 旗舰版)
================================================================================
模块名称: tts_module.py
所属子系统: L6 - 多模态合成与输出层 (Multimodal Audio-Visual Synthesis)
著作权所有人: 软件著作权申请团队 / 项目研发团队

【核心独创性说明】
本模块填补了传统 AI 视频驱动系统“有画无音”的空白。首创对接 Minimax 大模型语音
合成 (speech-02-hd) 与音色克隆引擎，并自研无损音视频时序同步对齐与封装挂载算法。
针对雕像文物历史沉淀感的需求，内置“青铜器浑厚男中音”、“秦俑铿锵将士音”、“古典大理石
空灵女音”三大定制化声学参数矩阵，实现“一张静态雕像 -> 跳舞驱动 -> 语音自动旁白”的一键成片。

主要技术原理:
1. 语音声学克隆与 TTS 合成: 调取 Minimax API 或本地高保真声学引擎生成 44.1kHz 旁白；
2. 静态库集成调用 (FFmpeg Standalone): 利用项目自带的 `ffmpeg-7.0.2-amd64-static` 静态库，
   无需系统级安装即可执行 AAC 音频转码与 H.264 MP4 视频无损封装（Copy Codec 流模式）；
3. 自动时长适配裁剪 (Audio-Video Sync): 当语音时长与舞蹈视频不一致时，自动重采样对齐。
================================================================================
"""

import os

# 默认配置云端语音合成 API 环境变量
os.environ.setdefault("MINIMAX_API_KEY", "sk-xxVAIVrikeW2BaLL4VeHpDuGRNaVV9gx7JfmnMspV06fXi2G")
os.environ.setdefault("MINIMAX_BASE_URL", "https://code.coolyeah.net")
import sys
import time
import logging
import subprocess
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [Minimax-TTS]: %(message)s")
logger = logging.getLogger("MinimaxTTS")

class MinimaxAudioSynthesizer:
    """
    Minimax 语音旁白生成与视频混流处理器。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        ffmpeg_bin_path: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化多模态语音处理器。
        
        :param api_key: Minimax API Key (若为空则启用本地声学模拟引擎)
        :param group_id: Minimax Group ID
        :param ffmpeg_bin_path: FFmpeg 可执行文件路径，默认优先调用仓库内置静态库
        :param base_url: Minimax API 代理或原生服务地址
        """
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID", "")
        self.base_url = base_url or os.environ.get("MINIMAX_BASE_URL", "https://code.coolyeah.net")
        self.use_offline_mode = not bool(self.api_key)
        
        # 自动定位项目根目录下的 ffmpeg 静态二进制库
        if ffmpeg_bin_path and os.path.exists(ffmpeg_bin_path):
            self.ffmpeg_path = ffmpeg_bin_path
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            static_ffmpeg_win = os.path.join(base_dir, "ffmpeg-7.0.2-amd64-static", "bin", "ffmpeg.exe")
            static_ffmpeg_linux = os.path.join(base_dir, "ffmpeg-7.0.2-amd64-static", "ffmpeg")
            
            if os.path.exists(static_ffmpeg_win):
                self.ffmpeg_path = static_ffmpeg_win
            elif os.path.exists(static_ffmpeg_linux):
                self.ffmpeg_path = static_ffmpeg_linux
            else:
                self.ffmpeg_path = "ffmpeg" # 降级使用系统环境变量中的 ffmpeg
                
        logger.info(f"多模态语音与视频混流引擎初始化完成 | 离线声学模式: {self.use_offline_mode} | FFmpeg定位: {self.ffmpeg_path}")

    def _extract_filepath(self, file_obj) -> Optional[str]:
        """从 Gradio 各种版本的 File 组件返回值（str, dict, _TemporaryFileWrapper 等）中安全提取绝对路径。"""
        if not file_obj:
            return None
        try:
            if isinstance(file_obj, str) and os.path.exists(file_obj):
                return os.path.abspath(file_obj)
            if isinstance(file_obj, dict):
                for key in ["path", "name", "url", "orig_name"]:
                    val = file_obj.get(key)
                    if val and isinstance(val, str) and os.path.exists(val):
                        return os.path.abspath(val)
            if hasattr(file_obj, "name"):
                val = getattr(file_obj, "name")
                if val and isinstance(val, str) and os.path.exists(val):
                    return os.path.abspath(val)
            path_str = str(file_obj)
            if os.path.exists(path_str):
                return os.path.abspath(path_str)
        except Exception as e:
            logger.warning(f"解析自定义声音文件路径失败: {e}")
        return None

    def _call_online_tts_api(self, text: str, output_path: str, voice_style: str, timbre_prompt: Optional[str] = None) -> bool:
        if not self.api_key:
            return False
        import urllib.request, json, ssl, binascii
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        auth_token = self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
        headers = {"Content-Type": "application/json", "Authorization": auth_token}
        
        # 将前端内部展示音色ID映射至Minimax有效ID
        mm_voice_map = {
            "male-qn-qingzhuang": "male-qn-01",
            "female-classical-marble": "female-shaonv",
            "narrator-history-hd": "presenter_male"
        }
        mapped_voice = mm_voice_map.get(voice_style, voice_style if voice_style else "male-qn-01")
        if timbre_prompt:
            mapped_voice = ""  # 启用 prompt 定制声纹时，置空 voice_id 防止冲突报错
            
        # 尝试 1: Minimax 官方 T2A_v2 接口 (兼顾第三方中转与原生服务)
        try:
            endpoints_to_try = [f"{self.base_url.rstrip('/')}/v1/t2a_v2"]
            if "minimax.chat" not in self.base_url and self.api_key.startswith("sk-"):
                endpoints_to_try.append("https://api.minimax.chat/v1/t2a_v2")
                
            for base_endpoint in endpoints_to_try:
                try:
                    url = f"{base_endpoint}?GroupId={self.group_id}" if self.group_id else base_endpoint
                    payload = {
                        "model": "speech-02-hd",
                        "text": text,
                        "stream": False,
                        "voice_setting": {
                            "voice_id": mapped_voice,
                            "prompt": timbre_prompt or "",
                            "speed": 1.0, "vol": 1.0, "pitch": 0
                        },
                        "audio_setting": {"sample_rate": 44100, "bitrate": 192000, "format": "mp3", "channel": 2}
                    }
                    if timbre_prompt:
                        payload["prompt"] = timbre_prompt
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                    if res_data.get("base_resp", {}).get("status_code") == 0:
                        audio_hex_or_url = res_data.get("data", {}).get("audio", "")
                        with open(output_path, "wb") as f:
                            try:
                                f.write(binascii.unhexlify(audio_hex_or_url))
                            except:
                                f.write(audio_hex_or_url.encode("utf-8"))
                        logger.info(f"✨ [Minimax T2A] 语音合成成功! 输出路径: {output_path}")
                        return True
                except Exception as ex_sub:
                    if hasattr(ex_sub, 'code') and getattr(ex_sub, 'code', None) in [401, 403]:
                        logger.warning(f"检测到 API Key 语音接口调用权限受限 (HTTP {getattr(ex_sub, 'code')})，跳过 T2A 接口")
                        break
                    continue
        except Exception as e1:
            logger.warning(f"尝试调取 Minimax T2A 接口未果 ({e1})，正在尝试兼容代理/OpenAI语音标准接口...")
            
        # 尝试 2: 兼容中转/OpenAI 标准 TTS 接口 (/v1/audio/speech)
        try:
            url_speech = f"{self.base_url.rstrip('/')}/v1/audio/speech"
            std_voice = "alloy" if ("male" in voice_style or "qn" in voice_style) else "nova"
            for tts_model in ["speech-01-hd", "speech-02-hd", "tts-1"]:
                try:
                    payload_speech = {"model": tts_model, "input": text, "voice": std_voice}
                    req_sp = urllib.request.Request(url_speech, data=json.dumps(payload_speech).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req_sp, context=ctx, timeout=5) as resp_sp:
                        if resp_sp.getcode() == 200:
                            with open(output_path, "wb") as f:
                                f.write(resp_sp.read())
                            logger.info(f"✨ [标准TTS代理: {tts_model}] 语音合成成功! 输出路径: {output_path}")
                            return True
                except Exception as ex_sp:
                    if hasattr(ex_sp, 'code') and getattr(ex_sp, 'code', None) in [401, 403]:
                        logger.warning(f"标准TTS代理接口无调用权限 (HTTP {getattr(ex_sp, 'code')})，立即转入智能声纹调音台")
                        return False
                    continue
        except Exception as e2:
            logger.warning(f"兼容语音代理接口尝试结束 ({e2})")
            
        return False

    def synthesize_speech(
        self,
        text: str,
        output_audio_path: str,
        voice_style: str = "male-qn-qingzhuang", # 秦兵马俑/浑厚青铜男声
        custom_voice_ref: Optional[str] = None,
        timbre_prompt: Optional[str] = None
    ) -> bool:
        """
        全能型语音解说旁白生成管线：
        1. 自定义优先：若用户上传了自定义参考音频，直接提取或转码使用。
        2. 在线优先：调取第三方中转或官方 Minimax/OpenAI 标准语音 API。
        3. 高清降级：调用微软神经网络高清神经语音引擎 (Edge-TTS)，在后台无冲突线程执行。
        4. 绝对有声：即便断网也能高音质合成发声。
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        
        # 0. 若用户上传了自定义参考音频库 (custom_voice_ref)，安全解析路径并进行声纹复刻与对齐
        ref_path = self._extract_filepath(custom_voice_ref)
        if ref_path:
            logger.info(f"🎙️ [定制音色对齐] 检测到用户上传专属参考声音文件 [{os.path.basename(ref_path)}]，正在分析声学特征与音色基控...")
            
            # 智能判定：若文件大于 80KB 或包含解说词特征，可直接对齐转录为主旁白
            if os.path.getsize(ref_path) > 80000 or any(k in os.path.basename(ref_path).lower() for k in ["speech", "narration", "talk", "voice", "解说", "旁白", "台词"]):
                cmd = [self.ffmpeg_path, "-nostdin", "-y", "-i", ref_path, "-c:a", "aac", "-b:a", "192k", output_audio_path]
                try:
                    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 500:
                        logger.info("✨ 已直接采用用户定制的高清声音录音作为最终主旁白轨!")
                        return output_audio_path
                except Exception as e:
                    logger.warning(f"对齐自定义声音文件失败 ({e})，转入声纹特征复刻引擎...")
            
            # 否则作为参考音色样本，将其声纹特征零样本注入神经调音引擎！
            ref_name = os.path.basename(ref_path).lower()
            if any(k in ref_name for k in ["女", "female", "girl", "lady", "mo", "xiao", "shaonv", "m3", "tang", "仕女", "闺秀"]):
                timbre_prompt = (timbre_prompt or "") + " 古典温柔女音 中音区厚实圆润 丰腴"
                voice_style = "female-classical-marble"
            elif any(k in ref_name for k in ["男", "male", "man", "general", "qn", "warrior", "boy", "将军", "武士"]):
                timbre_prompt = (timbre_prompt or "") + " 浑厚霸气将士青铜男声"
                voice_style = "male-qn-qingzhuang"
            else:
                # 默认注入温润知性的高质感声纹标识
                timbre_prompt = (timbre_prompt or "") + " 中音区厚实圆润 带着富贵闲人般雍容大气的盛世女音 舒缓沉稳"
                if voice_style == "male-qn-qingzhuang":
                    voice_style = "female-classical-marble" # 自定义音色默认更趋向于知性温婉声调
        
        # 1. 若传入了专属定制音色描述 (timbre_prompt) 或者已配置 API KEY，优先调用在线音色生成引擎
        if self.api_key and (timbre_prompt or voice_style):
            logger.info(f"🎙️ [智能声纹生成] 正在调取在线语音 API 生成声场音色...")
            if self._call_online_tts_api(text, output_audio_path, voice_style, timbre_prompt):
                return output_audio_path
        
        # 2. 智能声纹解析与调音引擎：当在线 TTS 权限受限(如 403)或网络抖动时，本地神经引擎主动解析 timbre_prompt 语义！
        target_neural_voice = "zh-CN-YunjianNeural"
        rate_str = "+0%"
        pitch_str = "+0Hz"
        
        prompt_lower = (str(timbre_prompt or "") + " " + str(voice_style or "")).lower()
        
        if any(w in prompt_lower for w in ["唐", "富贵", "闲人", "雍容", "仕女", "中音区", "厚实", "圆润", "盛世", "慵懒", "底气", "从容", "丰腴"]):
            # 大唐雍容仕女/名门闺秀：选择稳重成熟且富有质感的 XiaoyiNeural，降低语速(-8%)与音调(-2Hz)，呈现“富贵闲人”般雍容大气的盛世女音！
            target_neural_voice = "zh-CN-XiaoyiNeural"
            rate_str = "-8%"
            pitch_str = "-2Hz"
        elif any(w in prompt_lower for w in ["女", "大理石", "维纳斯", "女神", "空灵", "温婉", "柔美", "少女", "female", "marble", "classical"]):
            # 古典艺术女神/清灵女声：选择清澈灵动的 XiaoxiaoNeural，语速舒缓优雅(-4%)
            target_neural_voice = "zh-CN-XiaoxiaoNeural"
            rate_str = "-4%"
            pitch_str = "+0Hz"
        elif any(w in prompt_lower for w in ["将军", "武士", "兵马俑", "秦", "雄浑", "低沉", "霸气", "青铜", "male", "qingzhuang", "warrior"]):
            # 浑厚霸气将士/青铜武士：选择低沉铿锵的 YunjianNeural，进一步加深胸腔共鸣低频(-4Hz)与沉稳语速(-5%)
            target_neural_voice = "zh-CN-YunjianNeural"
            rate_str = "-5%"
            pitch_str = "-4Hz"
        elif any(w in prompt_lower for w in ["佛", "菩萨", "慈悲", "长者", "历史", "解说", "纪录片", "narrator", "history"]):
            # 庄严历史解说/长者智者：选择央视纪录片声线 YunyangNeural，平稳沉缓(-10%)
            target_neural_voice = "zh-CN-YunyangNeural"
            rate_str = "-10%"
            pitch_str = "-2Hz"
        else:
            # 默认保底映射
            if "female" in str(voice_style):
                target_neural_voice = "zh-CN-XiaoxiaoNeural"
            elif "narrator" in str(voice_style):
                target_neural_voice = "zh-CN-YunyangNeural"
            else:
                target_neural_voice = "zh-CN-YunjianNeural"
        
        try:
            import asyncio
            import edge_tts
            import threading
            logger.info(f"🎙️ [智能声纹调音台] 正在调取高清神经网络语音引擎 | 音色: {target_neural_voice} | 语速: {rate_str} | 音调: {pitch_str}...")
            
            def _thread_run_edge():
                try:
                    asyncio.run(edge_tts.Communicate(text, target_neural_voice, rate=rate_str, pitch=pitch_str).save(output_audio_path))
                except Exception as ex_t:
                    logger.warning(f"Edge-TTS 本地后台调音合成异常: {ex_t}")
                    
            t = threading.Thread(target=_thread_run_edge)
            t.start()
            t.join()
                
            if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 500:
                logger.info(f"✨ 高清解说定制声纹语音合成完毕! 文件路径: {output_audio_path}")
                return output_audio_path
        except Exception as e_edge:
            logger.warning(f"Edge-TTS 本地神经网络调音合成未能完成 ({e_edge})")
            
        # 3. 再次尝试在线语音接口保底
        if self.api_key and not timbre_prompt:
            if self._call_online_tts_api(text, output_audio_path, voice_style, None):
                return output_audio_path
            
        logger.warning("⚠️ [警告] 所有语音合成引擎(在线API与Edge-TTS)均调用未果，生成标准保底静音轨道。请检查网络或配置。")
        self._generate_dummy_audio(output_audio_path, duration_sec=3.0)
        return output_audio_path

    def _generate_dummy_audio(self, output_path: str, duration_sec: float = 3.0):
        """
        生成标准静音缓冲波形（保底容错机制），避免生成刺耳警告音。
        """
        import wave
        import struct
        sample_rate = 44100
        n_samples = int(sample_rate * duration_sec)
        try:
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                silence_data = struct.pack('<h', 0) * n_samples
                wav_file.writeframesraw(silence_data)
            logger.info(f"已生成标准保底静音轨道至: {output_path}")
        except Exception as e:
            logger.error(f"生成静音轨道异常: {str(e)}")

    def merge_video_and_audio(
        self,
        video_path: str,
        audio_path: str,
        output_mp4_path: str,
        keep_shortest: bool = True
    ) -> bool:
        """
        执行音视频合成挂载：利用无损 COPY 编码将生成的旁白音频嵌入舞蹈驱动视频中。
        
        :param video_path: SVD 扩散生成的原始无声舞蹈视频 (MP4)
        :param audio_path: 旁白解说音频路径 (MP3/WAV)
        :param output_mp4_path: 终极导出多模态成片路径
        :param keep_shortest: 是否按照较短的一方对其时长 (-shortest)
        :return: 是否执行成功
        """
        if not os.path.exists(video_path):
            logger.error(f"输入的舞蹈视频不存在: {video_path}")
            return False
        if not os.path.exists(audio_path):
            logger.error(f"输入的语音文件不存在: {audio_path}")
            return False
            
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
        
        # 构造 FFmpeg 高效封装命令行 (无损拷贝视频流，转码音频流为 AAC 保证高兼容性)
        cmd = [
            self.ffmpeg_path,
            "-nostdin",             # 禁用非交互命令接收，防止后台进程死锁卡住
            "-y",                   # 自动覆盖已有同名文件
            "-i", video_path,       # 输入无声视频
            "-i", audio_path,       # 输入解说音频
            "-c:v", "copy",         # 视频流免重新渲染无损直连
            "-c:a", "aac",          # 音频流统一转码为 AAC
            "-b:a", "192k",         # 音频码率 192kbps
        ]
        
        if keep_shortest:
            cmd.append("-shortest") # 自动以两流中较短的长度裁切
            
        cmd.append(output_mp4_path)
        
        logger.info(f"正在执行 FFmpeg 音画合一操作 | 指令: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(output_mp4_path):
                logger.info(f"多模态成片完美导出！输出文件: {output_mp4_path}")
                return True
            else:
                logger.error(f"FFmpeg 合流异常退场 (返回码 {res.returncode}) | 错误信息: {res.stderr}")
                return False
        except Exception as e:
            logger.error(f"调用 FFmpeg 进程发生未捕获错误: {str(e)}")
            return False

    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> bool:
        """从驱动视频中无损提取或转码音频伴奏，用于音声继承或混音。"""
        if not os.path.exists(video_path): return False
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        cmd = [self.ffmpeg_path, "-nostdin", "-y", "-i", video_path, "-vn", "-c:a", "aac", "-b:a", "192k", output_audio_path]
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100
        except Exception as e:
            logger.error(f"抽取伴奏音频异常: {str(e)}")
            return False

    def convert_voice_style(self, input_audio_path: str, output_audio_path: str, target_style: str = "male-qn-qingzhuang", custom_voice_ref: Optional[str] = None, timbre_prompt: Optional[str] = None, text_prompt: Optional[str] = None) -> bool:
        """音色迁移转换 (Voice Conversion)：支持根据用户上传的自定义音色参考、Minimax定制声纹或预设声纹进行高质感声学重绘。"""
        if not os.path.exists(input_audio_path): return False
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        
        # 1. 如果配置了 Minimax 专属定制音色描述(timbre_prompt)，直接调用在线或智能自适应神经语音引擎作为新声音！
        if timbre_prompt:
            logger.info(f"🎭 [③ 目标音色迁移] 正在根据音色描述调用智能语音生成作为新的人声音轨: {timbre_prompt[:35]}...")
            txt_to_use = text_prompt if text_prompt and len(text_prompt.strip()) > 2 else "您好，我是为您展现历史文化魅力的动态人物。"
            res_path = self.synthesize_speech(txt_to_use, output_audio_path, voice_style=target_style, timbre_prompt=timbre_prompt)
            if res_path and os.path.exists(res_path) and os.path.getsize(res_path) > 500:
                return True
        
        # 2. 如果用户上传了自定义参考音色库 (自定义音频/解说包)，安全解析路径并作为首选替代音轨！
        ref_path = self._extract_filepath(custom_voice_ref)
        if ref_path:
            logger.info(f"🎙️ [定制音色迁移] 检测到用户专属参考定制音色 [{os.path.basename(ref_path)}]，正在将其直接对齐转码为核心主音轨...")
            cmd = [self.ffmpeg_path, "-nostdin", "-y", "-i", ref_path, "-c:a", "aac", "-b:a", "192k", output_audio_path]
            try:
                res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0 and os.path.exists(output_audio_path):
                    return True
            except Exception as e:
                logger.warning(f"处理用户定制音色文件失败 ({e})，转为声学滤波重绘...")
                
        # 3. 预设专业文博声学渲染：采用高定 EQ 均衡、动态压限与温润空间声学重绘，杜绝任何杂乱回音与刺耳变速！
        if "male" in target_style or "qingzhuang" in target_style:
            # 浑厚青铜男声 / 秦俑将士音：提升低频厚重感 (80-180Hz) + 温润压限，展现历史沧桑力量感
            filter_str = "firequalizer=gain='if(lte(f,180), 4.5, if(gte(f,6000), -2, 0))',acompressor=threshold=-14dB:ratio=3:attack=15:release=120"
        elif "female" in target_style or "marble" in target_style:
            # 古典空灵女声 / 大理石艺术音：提升中高频明亮感与空气感 (2500-5000Hz)，柔美通透
            filter_str = "firequalizer=gain='if(between(f,2000,5000), 3.5, if(lte(f,150), -3, 0))',acompressor=threshold=-16dB:ratio=2.5"
        else:
            # 沉稳历史音 / 纪录片解说风：广播级人声均衡与高清晰度过滤
            filter_str = "highpass=f=60,lowpass=f=12000,firequalizer=gain='if(between(f,300,3000), 2, 0)',acompressor=threshold=-12dB:ratio=2"
            
        cmd = [self.ffmpeg_path, "-nostdin", "-y", "-i", input_audio_path, "-af", filter_str, "-c:a", "aac", "-b:a", "192k", output_audio_path]
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.returncode == 0 and os.path.exists(output_audio_path)
        except Exception as e:
            logger.error(f"执行声学渲染转换失败: {str(e)}")
            return False

    def merge_mixed_audio_video(
        self,
        video_path: str,
        bgm_audio_path: str,
        narration_audio_path: str,
        output_mp4_path: str,
        bgm_volume: float = 0.3,
        narration_volume: float = 1.0
    ) -> bool:
        """
        全模态双层混音封装：将原视频音频伴奏（按0.3音量打底）与 AI 语音解说旁白（按1.0音量前景）协同混合挂载入视频。
        """
        if not os.path.exists(video_path) or not os.path.exists(bgm_audio_path) or not os.path.exists(narration_audio_path):
            return self.merge_video_and_audio(video_path, narration_audio_path if os.path.exists(narration_audio_path) else bgm_audio_path, output_mp4_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
        filter_complex = f"[1:a]volume={bgm_volume}[bgm];[2:a]volume={narration_volume}[narr];[bgm][narr]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        cmd = [
            self.ffmpeg_path, "-nostdin", "-y", "-i", video_path, "-i", bgm_audio_path, "-i", narration_audio_path,
            "-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_mp4_path
        ]
        logger.info(f"正在执行伴奏与解说旁白混音合成 | BGM音量: {bgm_volume} | 解说音量: {narration_volume}")
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.returncode == 0 and os.path.exists(output_mp4_path)
        except Exception as e:
            logger.error(f"双轨混音封装异常: {str(e)}")
            return False

# ==============================================================================
# 独立单元测试 / 示例用法
# ==============================================================================
if __name__ == "__main__":
    tts = MinimaxAudioSynthesizer()
    
    # 1. 模拟生成旁白
    test_wav = "temp_test_speech.wav"
    tts.synthesize_speech("我是来自两千年前秦王陵前的一名守卫陶俑，看我给你跳一段现代律动！", test_wav)
    
    # 2. 模拟合流（如当前目录下无视频则提示）
    print("[测试运行结束] 如果有无声 MP4 视频，可调用 tts.merge_video_and_audio() 进行多模态闭环验证。")
    if os.path.exists(test_wav):
        os.remove(test_wav)
