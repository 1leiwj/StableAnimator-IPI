# -*- coding: utf-8 -*-
"""
AI 雕像动态驱动生成系统 - DeepSeek LLM 多模态智能体调度模块 (V3.0 旗舰版)
================================================================================
模块名称: agent.py
所属子系统: L2 - 智能调度层 (DeepSeek Agent Scheduler)
著作权所有人: 软件著作权申请团队 / 项目研发团队

【核心独创性说明】
本模块首创将大语言模型（DeepSeek-V3 / R1）深度集成至视频生成扩散模型的推理管线中。
通过定义 Function Calling 标准化工具接口，将用户的自然语言意图（例如“我想给一张青铜器
生成一段30秒的武术动作，显存只有16G”）自动转化为底层 SVD-XT 模型的最佳超参数与档位路由。

主要特性:
1. 意图解析与三档模式自动推导 (Mode Routing): 自动在“无IPI极速预览”、“IPI标准模式”、“HJB最优平滑”中择优；
2. 硬件显存保护推导 (OOM Guard Scheduler): 针对 <16GB 显存自动开启 Tile=16, Overlap=4 降维模式；
3. 离线规则双重保障机制 (Offline Rule Fallback): 在网络未连接或无 API Key 时自动降级为专业规则推理引擎。
================================================================================
"""

import os

# 默认配置云端大模型 API 环境变量
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-09ff9872f5dc432a992b320f915c4581")
os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [DeepSeek-Agent]: %(message)s")
logger = logging.getLogger("DeepSeekAgent")

class DeepSeekAgentScheduler:
    """
    DeepSeek 智能参数生成调度器，支持线上的 LLM Function Calling 与离线专家规则库双引擎。
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        """
        初始化智能体调度器。
        
        :param api_key: DeepSeek API Key（若未提供或环境变量中无，则自动启用离线专家规则引擎）
        :param base_url: DeepSeek 标准 REST API 端点
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.use_offline_mode = not bool(self.api_key)
        
        if self.use_offline_mode:
            logger.warning("未检测到有效 DEEPSEEK_API_KEY，系统已自动启用【离线专家规则调度引擎】")
        else:
            logger.info("已成功连接 DeepSeek LLM 接口，具备完整的在线自然语言意图推导能力")

    def _get_function_calling_schema(self) -> List[Dict[str, Any]]:
        """
        定义向底层扩散模型传递参数的 Function Calling Schema 规范。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "configure_stable_animator_pipeline",
                    "description": "根据用户输入的雕像类型、视频长度及硬件条件，配置 AI 雕像动态视频生成管线的最佳参数。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "inference_mode": {
                                "type": "integer",
                                "enum": [1, 2, 3],
                                "description": "1: 无IPI极速预览模式(仅PoseNet空间约束); 2: IPI双轨标准生成模式; 3: IPI+HJB最优轨迹平滑模式(最高质量)"
                            },
                            "tile_size": {
                                "type": "integer",
                                "enum": [8, 12, 16, 24],
                                "description": "滑动窗口去噪大小。16G及以下显存建议选16，大显存选24"
                            },
                            "frames_overlap": {
                                "type": "integer",
                                "default": 4,
                                "description": "相邻 Tile 分块间的重叠帧数，通常选 4 保证时序一致性"
                            },
                            "num_inference_steps": {
                                "type": "integer",
                                "default": 25,
                                "description": "去噪采样步数，推荐 20-30 步"
                            },
                            "guidance_scale": {
                                "type": "number",
                                "default": 4.5,
                                "description": "无分类器引导系数 (CFG)，非真人雕像推荐 4.5~5.5 增强材质约束"
                            },
                            "use_zero_face_embedding": {
                                "type": "boolean",
                                "default": True,
                                "description": "对于无真实人脸的雕像目标，必须开启 True 触发零向量降级策略"
                            },
                            "recommended_tts_prompt": {
                                "type": "string",
                                "description": "针对该雕像特质自动生成的符合历史背景的语音旁白文案（给 Minimax TTS 使用）"
                            }
                        },
                        "required": ["inference_mode", "tile_size", "num_inference_steps", "use_zero_face_embedding", "recommended_tts_prompt"]
                    }
                }
            }
        ]

    def _offline_rule_engine(self, user_prompt: str, vram_gb: int = 16) -> Dict[str, Any]:
        """
        离线专家规则引擎：当无网络或无 API Key 时，基于关键词匹配与显存探究提供专业推荐。
        """
        logger.info(f"执行离线规则调度 | 提示词: '{user_prompt}' | 目标显存: {vram_gb}GB")
        
        prompt_lower = user_prompt.lower()
        
        # 1. 模式判断 (Mode Selecting)
        if any(w in prompt_lower for w in ["极速", "预览", "快", "测试", "调试", "preview", "fast"]):
            mode = 1 # 极速预览
            steps = 15
        elif any(w in prompt_lower for w in ["平滑", "高质量", "最优", "专业", "最高", "丝滑", "抖动"]):
            mode = 3 # IPI + HJB 平滑
            steps = 28
        else:
            mode = 2 # 标准双轨
            steps = 25
            
        # 2. 显存策略与 Tile 设定
        if vram_gb <= 12:
            tile_size, overlap = 12, 3
        elif vram_gb <= 16:
            tile_size, overlap = 16, 4
        else:
            tile_size, overlap = 24, 6
            
        # 3. 目标属性挖掘 (雕像类型判别)
        is_statue = any(w in prompt_lower for w in ["雕像", "陶俑", "青铜", "兵马俑", "泥塑", "木雕", "佛像", "statue"])
        cfg_scale = 5.0 if is_statue else 4.2
        
        # 4. 自动匹配历史解说文案 (供给后续 TTS 模块)
        if "兵马俑" in prompt_lower or "秦" in prompt_lower:
            tts_text = "我乃大秦帝国陵前守卫陶俑。越千载风霜，今朝以数字之躯，为你呈现大秦雄风！"
        elif "青铜" in prompt_lower or "三星堆" in prompt_lower:
            tts_text = "沉睡数千年，一醒惊天下。见证蜀地古老文明的青铜之光，在此刻为之舞动。"
        elif "古希腊" in prompt_lower or "大理石" in prompt_lower:
            tts_text = "穿越时光的大理石艺术，来自古典美的永恒凝视，让我们用现代律动唤醒静态的永恒。"
        else:
            tts_text = "静态的艺术在此刻苏醒，数字 AI 引擎已为你开启跨越时空的奇迹共舞！"

        return {
            "status": "success",
            "source": "offline_rule_engine",
            "config": {
                "inference_mode": mode,
                "tile_size": tile_size,
                "frames_overlap": overlap,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "use_zero_face_embedding": True,
                "recommended_tts_prompt": tts_text
            },
            "reasoning": f"根据关键词解析，自动分配为模式 {mode}；针对 {vram_gb}GB 显存启用 Tile={tile_size} 优化机制。"
        }

    def schedule_pipeline(self, user_prompt: str, vram_gb: int = 16) -> Dict[str, Any]:
        """
        主对外调度接口：自动推导最匹配用户场景的生成参数配置包。
        
        :param user_prompt: 用户的自然语言描述（例如：“请生成一段三星堆青铜立人像的高质量太极舞蹈”）
        :param vram_gb: 当前机器可用的显存大小 (单位: GiB)
        :return: 包含推理模式、步数、Tile 大小、TTS 旁白文案的完整配置字典
        """
        if self.use_offline_mode:
            return self._offline_rule_engine(user_prompt, vram_gb)
            
        # 如果配置了 API Key，则尝试通过 requests 调用 DeepSeek 在线接口
        try:
            import urllib.request
            import ssl
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是 AI 雕像动态驱动生成系统（StableAnimator V3.0）的首席智能架构调度师。请根据用户需求调用 configure_stable_animator_pipeline 为底层扩散模型推导最优生成参数。"
                    },
                    {
                        "role": "user",
                        "content": f"系统显存限制: {vram_gb} GB. 用户需求说明: {user_prompt}"
                    }
                ],
                "tools": self._get_function_calling_schema(),
                "tool_choice": "auto",
                "temperature": 0.2
            }
            
            endpoint = f"{self.base_url.rstrip('/')}/chat/completions" if "chat/completions" not in self.base_url else self.base_url
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                
            tool_calls = resp_json["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                arguments_str = tool_calls[0]["function"]["arguments"]
                parsed_args = json.loads(arguments_str)
                logger.info(f"线上 DeepSeek Function Calling 解析成功: {parsed_args}")
                return {
                    "status": "success",
                    "source": "deepseek_llm_online",
                    "config": parsed_args,
                    "reasoning": resp_json["choices"][0]["message"].get("content", "DeepSeek 智能体推导完成")
                }
            else:
                logger.warning("LLM 未返回正确的 tool_calls，自动转入离线引擎补偿")
                return self._offline_rule_engine(user_prompt, vram_gb)
                
        except Exception as e:
            logger.error(f"线上 API 调用出错 ({str(e)})，触发自动降级保障，转入离线引擎")
            return self._offline_rule_engine(user_prompt, vram_gb)

    def analyze_character_timbre(self, image_path: Optional[str] = None, user_prompt: Optional[str] = None) -> Dict[str, str]:
        """
        DeepSeek 识图与文本分析：深度提取角色/文物的身份年代属性，生成 Minimax T2A 音色描述提示词 (Timbre Prompt) 与专属台词。
        """
        prompt_str = str(user_prompt or "")
        img_name = os.path.basename(image_path) if (image_path and isinstance(image_path, str)) else ""
        
        # 0. 本地图像视觉特征自动采样分析 (色调与材质分析，绝不预设性别偏见)
        visual_desc = ""
        visual_category = ""
        b64_img = ""
        if image_path and isinstance(image_path, str) and os.path.exists(image_path):
            try:
                import base64
                with open(image_path, "rb") as f_img:
                    b64_img = base64.b64encode(f_img.read()).decode("utf-8")
                
                from PIL import Image
                with Image.open(image_path) as img:
                    img_rgb = img.convert("RGB")
                    small_img = img_rgb.resize((32, 32))
                    pixels = list(small_img.getdata())
                    avg_r = sum(p[0] for p in pixels) / len(pixels)
                    avg_g = sum(p[1] for p in pixels) / len(pixels)
                    avg_b = sum(p[2] for p in pixels) / len(pixels)
                    brightness = (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114)
                    
                    if brightness > 165 and abs(avg_r - avg_g) < 25 and abs(avg_g - avg_b) < 25:
                        visual_category = "marble"
                        visual_desc = f"【图像视觉色彩采样】: 明亮冷灰/大理石白底色(亮度:{brightness:.1f})，具有古典雕塑或石雕造像材质特征。"
                    elif avg_r > 120 and avg_g > 100 and avg_b < 100:
                        visual_category = "buddha"
                        visual_desc = f"【图像视觉色彩采样】: 暖黄/石窟泥金底色(暖黄饱和度高)，具有东方石窟古典佛像或菩萨造像特征。"
                    elif avg_r > 70 and abs(avg_r - avg_g) < 35 and avg_b < avg_r:
                        visual_category = "bronze"
                        visual_desc = f"【图像视觉色彩采样】: 沉稳陶土/青铜石雕底色(亮度:{brightness:.1f})，具有陶俑、石雕、青铜造像等文博历史材质特征。"
                    else:
                        visual_desc = f"【图像视觉色彩采样】: 均衡色彩结构(R:{avg_r:.0f},G:{avg_g:.0f},B:{avg_b:.0f})，具备历史文物与艺术雕塑质感。"
            except Exception as ex:
                logger.debug(f"本地图像视觉特征采样常规提示: {ex}")

        combined_info = (prompt_str + " " + img_name + " " + visual_category + " " + visual_desc).lower()
        
        # 1. 线上多模态大模型识图智能分析 (传递真正的 Base64 图像与性别识别指令)
        if not self.use_offline_mode and self.api_key:
            try:
                import urllib.request, ssl, json
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                sys_msg = (
                    "你是一个顶级文博声学总监、艺术鉴赏家与AI音色架构师。请仔细观察用户附加的图片或文物描述（【核心要领】：务必像深度艺术鉴赏一样，仔细观察造像的朝代特征、体态神态、服饰气度与角色性别！例如若是唐代仕女/女俑，务必鉴赏其丰腴圆润、神态雍容自信与‘富贵闲人’的盛世气度；切勿把女性误判为男性或陵墓将士！），只返回标准的 JSON 格式结果，包含四个字段：\n"
                    "1. 'character_name': 准确的角色与朝代美学身份认定（例如'唐代加彩陶仕女俑 / 盛世大唐雍容女俑' / '汉代彩绘仕女' / '古希腊优雅艺术女神'等）\n"
                    "2. 'timbre_prompt': 专门给 Minimax T2A 语音模型使用的音色气质量身定制描述（要求60字左右。特别注意：如果是唐代仕女或古代贵族女俑，声纹绝不能是娇滴滴的少女，也不应是苍老威严的老妪，而应描述为【中音区饱满温润、语速沉稳舒缓，带着‘富贵闲人’般雍容大气、略带慵懒却充满底气的盛世从容女声】；若是神话女神，则通透灵动；若是青铜将军，则低沉浑厚）\n"
                    "3. 'recommended_tts_prompt': 适合该角色口吻与神态的解说旁白（60字左右。需契合人物神态，如仕女要体现盛世自信从容，严禁硬汉用词！）\n"
                    "4. 'voice_style': 推荐的基本声纹映射ID（女性必须返回 'female-classical-marble'；男性将军武士返回 'male-qn-qingzhuang'；长者或佛像返回 'narrator-history-hd'）"
                )
                
                # 若连接的是 coolyeah 或 minimax 中转，自动适配支持视觉的多模态模型名称
                target_model = "MiniMax-M3" if ("coolyeah" in self.base_url or "minimax" in self.base_url.lower()) else "deepseek-chat"
                
                user_msg_content = [
                    {"type": "text", "text": f"分析对象名称/描述/文件名与图像视觉采样:\n{prompt_str or '未提供文字说明'}\n文件名: {img_name}\n{visual_desc}\n【鉴赏与声学引导】：请仔细观察附图中的人物姿态与朝代气质。若是仕女俑或女性雕塑，务必揣摩其神态气度（如唐俑的雍容自信、富贵从容），定制符合气质的高维音色提示词与旁白！"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ] if b64_img else f"分析对象名称/描述/文件名与图像视觉采样:\n{prompt_str or '未提供文字说明'}\n文件名: {img_name}\n{visual_desc}\n【鉴赏与声学引导】：请揣摩人物造像气度，定制符合其历史格局与身份气质的音色提示词与旁白！"
                
                payload = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg_content}
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"}
                }
                endpoint = f"{self.base_url.rstrip('/')}/chat/completions" if "chat/completions" not in self.base_url else self.base_url
                req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, context=ctx, timeout=12) as response:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    content_str = resp_json["choices"][0]["message"].get("content", "{}")
                    result = json.loads(content_str)
                    if result.get("timbre_prompt") and result.get("recommended_tts_prompt"):
                        logger.info(f"✨ 线上多模态识图音色分析成功 (模型: {target_model}): {result}")
                        return result
            except Exception as e:
                logger.warning(f"线上多模态识图分析回调受挫 ({e})，启用离线专家声学分析规则库...")
                
        # 2. 离线专家声学分析规则引擎 (对标 DeepSeek 网页版高阶审美：精细区分大唐仕女、西方女神与各类文博雕像)
        tang_lady_keywords = ["唐", "仕女", "侍女", "陶女", "女俑", "陶俑", "贵族", "闺秀", "宫", "妃", "后", "娘", "彩绘", "tang", "lady", "beauty", "siren"]
        if any(w in combined_info for w in tang_lady_keywords):
            return {
                "character_name": "唐代加彩陶仕女俑 / 盛世大唐雍容女俑",
                "timbre_prompt": "声线饱满温润、中音区厚实圆润，既非娇滴滴的少女亦非苍老的老妪，而是带着‘富贵闲人’般雍容大气、略带慵懒却充满底气的盛世从容女音，语速舒缓沉稳、吐字清晰自信。",
                "recommended_tts_prompt": "体态丰腴，神态雍容。见证大唐盛世的从容与底气。今日我便以这数字光影之躯，为你展现跨越千年的大唐风华。",
                "voice_style": "female-classical-marble"
            }
        elif visual_category == "marble" or any(w in combined_info for w in ["女", "大理石", "维纳斯", "希腊", "女神", "marble", "female", "woman", "girl", "princess", "goddess"]):
            return {
                "character_name": "古典女性石雕造像 / 优雅艺术女神",
                "timbre_prompt": "声线优雅温婉、清澈灵动，中高频明亮通透，发音轻盈柔美、语速舒缓高雅，带有古典东方美学与艺术造像的空气感，宛如穿越千年的晨曦回响。",
                "recommended_tts_prompt": "跨越千年的时光凝视，见证古典雕塑的优美与灵动。让我们用现代的韵律，唤醒沉静造像中温婉隽永的生命诗篇。",
                "voice_style": "female-classical-marble"
            }
        elif any(w in combined_info for w in ["三星堆", "立人", "蜀", "bronze"]):
            return {
                "character_name": "三星堆古蜀文明青铜大立人像",
                "timbre_prompt": "极具金属质感与胸腔共鸣的浑厚男中音，低频饱满深沉，发音吐字带有远古祭祀的庄严与神秘感，语速沉稳有力、回响绵长。",
                "recommended_tts_prompt": "沉睡数千年，一醒惊天下。见证古蜀文明的青铜之光，在此刻为跨越三千年的交汇而舞动。",
                "voice_style": "male-qn-qingzhuang"
            }
        elif visual_category == "buddha" or any(w in combined_info for w in ["佛", "菩萨", "金刚", "罗汉", "石窟", "buddha", "观音"]):
            return {
                "character_name": "东方石窟古典造像 / 慈眉善目菩萨像",
                "timbre_prompt": "温润慈祥、中正平稳的庄严历史音色，声线宽厚包容，带有寺庙大殿与石窟空间的微弱混响，语调平和慈悲、安宁静谧。",
                "recommended_tts_prompt": "历经千载风沙，眉目依旧慈悲。当数字光影照进千年石窟，让沉静的艺术展现灵动的梵音律动。",
                "voice_style": "narrator-history-hd"
            }
        else: # 默认历史武士 / 将军像
            return {
                "character_name": "大秦陵前青铜武士俑 / 历史将士造像",
                "timbre_prompt": "声音低沉有力、胸腔共鸣浑厚，带有地下陵寝青铜金属的碰撞质感和千年沧桑的历史威严，咬字铿锵有力、语速沉稳霸气。",
                "recommended_tts_prompt": "我乃大秦帝国陵前守卫武士！越千载风霜，今朝以数字之躯，为你呈现大秦将士的气吞万里如虎！",
                "voice_style": "male-qn-qingzhuang"
            }


# ==============================================================================
# 独立单元测试 / 示例用法
# ==============================================================================
if __name__ == "__main__":
    scheduler = DeepSeekAgentScheduler() # 默认测试离线专家规则
    
    test_prompts = [
        "我想生成一段秦始皇兵马俑的高质量跳街舞视频，希望能丝滑去抖动！",
        "快，简单给个大理石古希腊雕像的测试预览，用最快速度跑出来看看效果。"
    ]
    
    for p in test_prompts:
        print("\n" + "="*70)
        print(f"[测试用户需求]: {p}")
        res = scheduler.schedule_pipeline(p, vram_gb=16)
        print(f"[智能调度输出]: {json.dumps(res['config'], indent=2, ensure_ascii=False)}")
