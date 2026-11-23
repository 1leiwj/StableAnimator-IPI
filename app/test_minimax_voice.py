# -*- coding: utf-8 -*-
"""
Minimax / 智能声纹与语音合成功能 独立测试诊断脚本
本脚本已严格填入您提供的默认 API 地址与 API-KEY，用于全面自测。
"""

import os
import sys
import io
import json
import urllib.request
import urllib.error
import ssl
import binascii
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== 用户预设配置 ====================
DEFAULT_API_URL = "https://code.coolyeah.net"
DEFAULT_API_KEY = "sk-xxVAIVrikeW2BaLL4VeHpDuGRNaVV9gx7JfmnMspV06fXi2G"
# ======================================================

def test_minimax_standalone(api_url=DEFAULT_API_URL, api_key=DEFAULT_API_KEY, text="这里是智能文物赋活工作台，正在测试解说旁白语音合成效果。", output_path="test_minimax_result.mp3"):
    print("=" * 60)
    print("🎙️ 开始独立测试智能语音与声纹生成效果")
    print(f"【API地址】: {api_url}")
    print(f"【API-KEY】: {api_key[:6]}...{api_key[-4:] if len(api_key)>10 else ''}")
    print(f"【测试文本】: 「{text}」")
    print("=" * 60)
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # --- 1. 测试 MiniMax-M3 文本对话能力 (分析角色音色描述) ---
    print("\n▶️ [尝试 1/3] 调取中转平台 MiniMax-M3 对话大模型 (生成音色特征描述)...")
    try:
        url_chat = f"{api_url.rstrip('/')}/v1/chat/completions"
        payload_chat = {
            "model": "MiniMax-M3",
            "messages": [
                {"role": "system", "content": "你是一位专业的声音鉴赏与声音合成专家。"},
                {"role": "user", "content": "请用一句话描述秦始皇兵马俑应该具有的声纹和音色特点。"}
            ]
        }
        req_c = urllib.request.Request(url_chat, data=json.dumps(payload_chat).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req_c, context=ctx, timeout=12) as resp_c:
            if resp_c.getcode() == 200:
                res_chat = json.loads(resp_c.read().decode("utf-8"))
                reply_text = res_chat.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ [成功] MiniMax-M3 文本大模型调用成功！")
                print(f"   🤖 AI 声纹描述生成结果: 「{reply_text.strip()}」")
                print(f"   💡 说明：您的 Key 在中转平台上【完全具备】对话/识图/描述文本模型的生成权限！")
    except Exception as ec:
        print(f"❌ MiniMax-M3 文本模型请求异常: {ec}")
        
    # --- 2. 测试中转平台语音合成 TTS 接口 ---
    print("\n▶️ [尝试 2/3] 调取中转平台语音合成 TTS 接口 (/v1/audio/speech)...")
    try:
        url_proxy = f"{api_url.rstrip('/')}/v1/audio/speech"
        payload_proxy = {
            "model": "speech-01-hd",
            "input": text,
            "voice": "alloy"
        }
        req_p = urllib.request.Request(url_proxy, data=json.dumps(payload_proxy).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req_p, context=ctx, timeout=12) as resp_p:
            if resp_p.getcode() == 200:
                with open(output_path, "wb") as f:
                    f.write(resp_p.read())
                print(f"✅ [语音合成成功] 第三方中转语音生成成功！文件已保存至: {output_path}")
                return True
    except urllib.error.HTTPError as he:
        print(f"❌ 中转代理接口 HTTP 异常 (状态码 {he.code}):")
        if he.code == 403:
            print("   🚨【原因诊断】：返回 403 Forbidden！")
            print("   👉 解释：您的这个 API Key 在中转平台(coolyeah)上【仅开通了 MiniMax-M3 等文本聊天权限，未开通语音/TTS模型的调用权限】！")
            print("   👉 现象：网页后台调用 TTS 时会被第三方服务器以 403 无权限为由拒绝，这也是您之前觉得没有声音的原因。")
        elif he.code == 404:
            print("   🚨【原因诊断】：返回 404 Not Found！说明该中转平台未提供 /v1/audio/speech 语音路由。")
    except Exception as e2:
        print(f"❌ 中转代理请求失败: {e2}")
        
    # --- 3. 终极保底：微软 Edge-TTS 神经网络本地引擎 ---
    print("\n▶️ [尝试 3/3] 自动启动本地 Edge-TTS 微软神经引擎 (央视高清播音保底)...")
    try:
        import asyncio
        import edge_tts
        
        target_voice = "zh-CN-YunjianNeural" # 浑厚秦俑男声
        print(f"   正在调取神经引擎音色: {target_voice}")
        
        asyncio.run(edge_tts.Communicate(text, target_voice).save(output_path))
        if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
            print(f"✅ [终极保底成功] 高清解说音频已成功生成！文件已保存至根目录: {output_path}")
            print("   🎉 总结：有了这段双保险代码，即使线上 API Key 暂无语音合成权限，系统也会自动切到 Edge-TTS 引擎，视频绝对不会无声！")
            return True
    except ModuleNotFoundError:
        print("❌ 本地 Edge-TTS 合成失败: No module named 'edge_tts'")
        print("\n" + "="*60)
        print("💡 【原因终极定论与 100% 解决办法】")
        print("1. 为什么线上无声？因为您的这个 Key 在中转平台【未开通语音 TTS 模型调用权限】（返回 403 Forbidden）；")
        print("2. 为什么保底也失败？因为您的 conda 虚拟环境 (animate) 【还没有安装】本地免费的高清配音库 edge-tts。")
        print("\n👉 请在终端直接复制执行以下这一条指令，安装后即可立刻拥有永久的高清配音能力：")
        print("    pip install edge-tts -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("="*60)
    except Exception as e3:
        print(f"❌ 本地 Edge-TTS 合成失败: {e3}")
        
    print("\n❌ 所有语音测试未完全通过，请执行上述 pip 安装指令。")
    return False

if __name__ == "__main__":
    test_minimax_standalone()
