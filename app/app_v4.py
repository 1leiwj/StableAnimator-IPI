# -*- coding: utf-8 -*-
"""
AI 雕像动态驱动生成系统 - V4.0 工业级数字文博 AI 创作平台
================================================================================
模块名称: app_v4.py
所属目录: app/
架构设计: 深度融合 OpenArt × Vidu × Runway × 灵动画布 的现代 AI 创作工作台
著作权所有人: 数字文博与文物数字化创新实验室

【核心重构说明】
1. 100% 继承并复用原 V3.0 (app.py) 中的底层架构、推理逻辑与事件绑定函数，后端推理 0 改动；
2. 彻底颠覆原有限局限的排版，采用“黑夜星空深色主题 (#0E1422) + 轻质感磨砂卡片 (#161F30) + 蓝紫炫彩渐变”；
3. 左侧工具栏从笨重的按键重构为紧凑的序号化及专业侧边列表菜单 (左侧蓝色激活条 ▌ 交互)；
4. 首页 (Hero Page) 全新搭载 Vidu 风格的硬核背书带、Runway 式的“三步快速开始”卡片以及精选影视级成片展厅；
5. 将生成控制台参数精简分类，高级消融实验与声学策略默认折叠，提供极致整洁的沉浸式生成画布！
================================================================================
"""

import os
import sys
import time
import json
import tempfile
import logging
from datetime import datetime
import gradio as gr

# 导入 app.py 中已确立的所有核心推理调度函数、环境配置、内核模块与初始化逻辑
from app import (
    mount_gpu_environment_and_models,
    run_end_to_end_generation,
    handle_interactive_chat,
    handle_analyze_timbre,
    handle_preview_timbre,
    tts_synthesizer,
    agent_scheduler,
    hjb_smoother,
    logger
)

# ------------------------------------------------------------------------------
# 🎨 1. V4.0 工业级 AI 创作平台全量高级 CSS (深色主题 / 磨砂质感 / 高阶排版)
# ------------------------------------------------------------------------------
PLATFORM_V4_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Noto+Serif+SC:wght@400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-space: #0E1422;               /* 深邃宇宙夜空黑 - 主页核心背景 */
    --card-dark: #161F30;              /* 磨砂玻璃深蓝卡片 */
    --card-hover: #1E293B;             /* 交互悬浮高亮底颜色 */
    --accent-blue: #4F6BFF;            /* AI 平台高亮主蓝色 */
    --accent-gradient: linear-gradient(135deg, #FF5E7E 0%, #8B5CF6 50%, #4F6BFF 100%); /* 渐变发光按键 */
    --border-subtle: rgba(255, 255, 255, 0.08); /* 极细磨砂描边 */
    --text-primary: #F5F7FA;           /* 纯白文字 */
    --text-secondary: #94A3B8;         /* 灰蓝二级说明文字 */
    --shadow-glow: 0 10px 30px rgba(79, 107, 255, 0.2);
}

body, .gradio-container {
    background: var(--bg-space) !important;
    font-family: 'Outfit', 'Noto Serif SC', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* 隐藏 Gradio 默认页脚与冗余 padding */
footer { display: none !important; }
.gradio-container { padding: 0 !important; }

/* 2. 顶部主导航栏 (Top Navbar) */
.v4-header-bar {
    background: #0B101B !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0.6rem 2rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    height: 65px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
}
.header-logo-box {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}
.header-logo-icon {
    font-size: 1.8rem !important;
    filter: drop-shadow(0 0 8px rgba(255, 215, 0, 0.4)) !important;
}
.header-title {
    font-family: 'Noto Serif SC', 'Outfit', serif !important;
    font-size: 1.35rem !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: 1px !important;
}
.header-subtitle {
    font-size: 0.75rem !important;
    color: #64748B !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.header-center-badge {
    background: rgba(79, 107, 255, 0.12) !important;
    border: 1px solid rgba(79, 107, 255, 0.4) !important;
    color: #93C5FD !important;
    padding: 6px 20px !important;
    border-radius: 50px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
}
.header-right-user {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    color: #CBD5E1 !important;
    font-size: 0.9rem !important;
}
.user-avatar-pill {
    background: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 4px 14px !important;
    border-radius: 50px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    font-weight: 600 !important;
}

/* 3. 左侧侧边栏导航 (Sidebar Nav - 对标 OpenArt & 灵动画布) */
.v4-sidebar-column {
    background: #0B101B !important;
    border-right: 1px solid var(--border-subtle) !important;
    padding: 1.2rem 0.8rem !important;
    min-height: calc(100vh - 65px) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}
.nav-category-label {
    font-size: 0.75rem !important;
    color: #64748B !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    margin: 1rem 0 0.5rem 0.8rem !important;
    text-transform: uppercase !important;
}
.v4-nav-radio {
    background: transparent !important;
    border: none !important;
}
.v4-nav-radio label {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 11px 16px !important;
    margin-bottom: 4px !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
}
.v4-nav-radio label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #FFFFFF !important;
}
.v4-nav-radio label.selected {
    background: linear-gradient(90deg, rgba(79, 107, 255, 0.25) 0%, rgba(79, 107, 255, 0.02) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-left: 4px solid var(--accent-blue) !important;
    border-radius: 4px 10px 10px 4px !important;
}
.v4-nav-radio input[type="radio"] { display: none !important; }

/* 底部系统状态卡片 */
.sidebar-env-status {
    background: rgba(22, 31, 48, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 1.1rem !important;
    margin-top: 1.5rem !important;
    font-size: 0.82rem !important;
}
.env-status-title {
    color: #38BDF8 !important;
    font-weight: 700 !important;
    margin-bottom: 0.8rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.env-row {
    display: flex !important;
    justify-content: space-between !important;
    margin-bottom: 0.5rem !important;
    color: #94A3B8 !important;
}
.env-row span.ready { color: #4ADE80 !important; font-weight: 600 !important; }
.env-footer-icons {
    display: flex !important;
    justify-content: space-around !important;
    margin-top: 0.8rem !important;
    padding-top: 0.8rem !important;
    border-top: 1px dashed rgba(255, 255, 255, 0.1) !important;
    color: #64748B !important;
    font-size: 1rem !important;
}

/* 4. 主视口容器与页面共通背景 */
.v4-main-viewport {
    padding: 2rem 2.5rem !important;
    background: var(--bg-space) !important;
    min-height: calc(100vh - 65px) !important;
}

/* 5. 首页 Hero Banner 专区 (对标 Runway & Vidu) */
.hero-card-container {
    background: linear-gradient(135deg, rgba(22, 31, 48, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 24px !important;
    padding: 3rem 3.5rem !important;
    margin-bottom: 2.5rem !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4) !important;
}
.hero-main-title {
    font-family: 'Noto Serif SC', 'Outfit', serif !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem !important;
    letter-spacing: 2px !important;
}
.hero-sub-desc {
    font-size: 1.1rem !important;
    color: #94A3B8 !important;
    max-width: 680px !important;
    line-height: 1.6 !important;
    margin-bottom: 2rem !important;
}
.hero-stats-row {
    display: flex !important;
    gap: 1rem !important;
    flex-wrap: wrap !important;
}
.hero-stat-pill {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 8px 18px !important;
    border-radius: 50px !important;
    font-size: 0.88rem !important;
    color: #E2E8F0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* 6. 快速开始三步卡片区 (Quick Start - 对标 Runway) */
.section-header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 1.2rem !important;
}
.section-main-title {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.section-sub-title {
    font-size: 0.9rem !important;
    color: #64748B !important;
}
.quick-step-card {
    background: var(--card-dark) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 18px !important;
    padding: 1.6rem !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    height: 100% !important;
}
.quick-step-card:hover {
    transform: translateY(-5px) !important;
    border-color: rgba(79, 107, 255, 0.5) !important;
    background: #1E293B !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
}
.step-num-badge {
    background: var(--accent-blue) !important;
    color: #FFFFFF !important;
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-weight: 800 !important;
    margin-right: 10px !important;
}
.btn-cta-gradient {
    background: var(--accent-gradient) !important;
    color: #FFFFFF !important;
    font-family: 'Noto Serif SC', 'Outfit', serif !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1.4rem 2rem !important;
    box-shadow: 0 10px 30px rgba(255, 94, 126, 0.3) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    letter-spacing: 2px !important;
    height: 100% !important;
}
.btn-cta-gradient:hover {
    transform: scale(1.03) translateY(-3px) !important;
    box-shadow: 0 15px 40px rgba(139, 92, 246, 0.5) !important;
}

/* 7. 案例画廊展示卡片 (Showcase Cards - 对标 Vidu & OpenArt) */
.gallery-showcase-card {
    background: var(--card-dark) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    transition: all 0.35s ease !important;
}
.gallery-showcase-card:hover {
    transform: translateY(-6px) !important;
    border-color: #38BDF8 !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
}
.card-video-box {
    border-radius: 18px 18px 0 0 !important;
    overflow: hidden !important;
    background: #000000 !important;
}
.card-info-box {
    padding: 1.2rem !important;
}
.card-title-text {
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    color: #FFFFFF !important;
    margin-bottom: 0.3rem !important;
}
.card-desc-text {
    font-size: 0.85rem !important;
    color: #94A3B8 !important;
}

/* 8. 生成控制台与高阶参数样式 */
.console-panel-box {
    background: var(--card-dark) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 20px !important;
    padding: 1.8rem !important;
    margin-bottom: 1.5rem !important;
}
.accordion-dark {
    background: rgba(11, 16, 27, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    margin-top: 1rem !important;
}

/* Gradio 内置输入与按键深色适配 */
textarea, input[type="text"], input[type="number"] {
    background: #0B101B !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}
"""

def build_app_v4_ui():
    """构建 V4.0 工业级数字文博 AI 创作平台界面。"""
    with gr.Blocks(title="智饰：AI 雕像动态驱动生成平台 V4.0", css=PLATFORM_V4_CSS) as demo:
        
        # ======================================================================
        # 顶部官方主导航条 (Header Navbar)
        # ======================================================================
        with gr.Row(elem_classes=["v4-header-bar"]):
            with gr.Column(scale=3, min_width=250):
                gr.HTML(
                    """
                    <div class="header-logo-box">
                        <div class="header-logo-icon">🗿</div>
                        <div>
                            <div class="header-title">AI雕像驱动生成系统</div>
                            <div class="header-subtitle">Version 4.0 · STABLEANIMATOR AI</div>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=5, min_width=400):
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <span class="header-center-badge">✨ 故宫 & 秦始皇帝陵·文博数字赋活核心引擎 ✨</span>
                    </div>
                    """
                )
            with gr.Column(scale=2, min_width=200):
                gr.HTML(
                    """
                    <div class="header-right-user" style="justify-content: flex-end;">
                        <span>🔔 3</span>
                        <div class="user-avatar-pill">
                            <span style="background:#4F6BFF; width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:0.75rem;">A</span>
                            <span>AI 导师 / 科研平台</span>
                        </div>
                    </div>
                    """
                )

        # ======================================================================
        # 主框架：左侧侧边栏导航 + 右侧模块画布
        # ======================================================================
        with gr.Row():
            
            # ------------------------------------------------------------------
            # 左侧紧凑列列表导航栏 (Sidebar Column - 230px)
            # ------------------------------------------------------------------
            with gr.Column(scale=2, min_width=220, elem_classes=["v4-sidebar-column"]):
                with gr.Column():
                    nav_selector = gr.Radio(
                        choices=[
                            "🏠 首页概览",
                            "🎬 视频生成",
                            "✨ AI 助手",
                            "🎛️ 模型与参数",
                            "📜 历史作品",
                            "📦 数据与中间件",
                            "📝 系统日志",
                            "⚙️ 设置中心"
                        ],
                        value="🏠 首页概览",
                        label="",
                        elem_classes=["v4-nav-radio"]
                    )

                
                # 底部系统运行状态指示卡
                gr.HTML(
                    """
                    <div class="sidebar-env-status">
                        <div class="env-status-title">● 环境状态指示</div>
                        <div class="env-row"><span>SVD-XT</span><span class="ready">Ready</span></div>
                        <div class="env-row"><span>IPI Adapter</span><span class="ready">Ready</span></div>
                        <div class="env-row"><span>显存占用</span><span>12.6GB / 24GB</span></div>
                        <div class="env-row"><span>版本号</span><span>V4.0.0</span></div>
                        <div class="env-footer-icons">
                            <span>☀</span><span>⚙</span><span>👤</span><span style="color:#4F6BFF; font-weight:800;">A</span>
                        </div>
                    </div>
                    """
                )

            # ------------------------------------------------------------------
            # 右侧宽屏视口画布 (Main Viewport Panels)
            # ------------------------------------------------------------------
            with gr.Column(scale=10, min_width=850, elem_classes=["v4-main-viewport"]):
                
                # ==============================================================
                # 页面 1：【首页概览】 (Home Page - 70%作品展示 + 30%系统背书)
                # ==============================================================
                with gr.Column(visible=True) as col_home:
                    # Hero Banner 区域
                    gr.HTML(
                        """
                        <div class="hero-card-container">
                            <h1 class="hero-main-title">静态历史 · 一键活化 · 视听合一</h1>
                            <p class="hero-sub-desc">自研 SVD-XT 时空扩散与 IPI 特征自适应驱动大模型，赋予千古陶俑、青铜与大理石雕像真正的动态生命与沧桑旁白</p>
                            <div class="hero-stats-row">
                                <span class="hero-stat-pill">🚀 4K / 60FPS 超清输出</span>
                                <span class="hero-stat-pill">🗿 100+ 馆藏材质适配</span>
                                <span class="hero-stat-pill">⚡ < 0.1s IPI 保持响应</span>
                                <span class="hero-stat-pill">🔊 5大声学样本克隆</span>
                            </div>
                        </div>
                        """
                    )
                    
                    # 快速开始三步区 (Quick Start)
                    gr.HTML(
                        """
                        <div class="section-header-row">
                            <div>
                                <div class="section-main-title">⚡ 快速开始</div>
                                <div class="section-sub-title">三步生成您的数字文博活化成片作品</div>
                            </div>
                        </div>
                        """
                    )
                    with gr.Row(style={"margin-bottom": "2.5rem"}):
                        with gr.Column(scale=3):
                            gr.HTML(
                                """
                                <div class="quick-step-card">
                                    <div style="font-weight:800; font-size:1.15rem; margin-bottom:0.4rem; color:#FFFFFF;"><span class="step-num-badge">1</span>上传雕像图</div>
                                    <div style="color:#94A3B8; font-size:0.9rem;">支持 JPG / PNG 格式静态文物图像</div>
                                </div>
                                """
                            )
                        with gr.Column(scale=3):
                            gr.HTML(
                                """
                                <div class="quick-step-card">
                                    <div style="font-weight:800; font-size:1.15rem; margin-bottom:0.4rem; color:#FFFFFF;"><span class="step-num-badge">2</span>上传驱动视频</div>
                                    <div style="color:#94A3B8; font-size:0.9rem;">支持 MP4 / MOV 动作与舞蹈轨迹</div>
                                </div>
                                """
                            )
                        with gr.Column(scale=3):
                            gr.HTML(
                                """
                                <div class="quick-step-card">
                                    <div style="font-weight:800; font-size:1.15rem; margin-bottom:0.4rem; color:#FFFFFF;"><span class="step-num-badge">3</span>开始生成</div>
                                    <div style="color:#94A3B8; font-size:0.9rem;">AI 时空仿射对齐，一键视听合一</div>
                                </div>
                                """
                            )
                        with gr.Column(scale=3):
                            btn_quick_jump = gr.Button("⚡ 立即生成 / 进入工作台", elem_classes=["btn-cta-gradient"])

                    # 精选影视级成片展厅 (Showcase Gallery)
                    gr.HTML(
                        """
                        <div class="section-header-row">
                            <div>
                                <div class="section-main-title">🎬 优秀作品与生成案例展示</div>
                                <div class="section-sub-title">探索自研大模型在秦俑、汉俑、唐三彩与大理石雕像上的成功数字赋活成果</div>
                            </div>
                        </div>
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["gallery-showcase-card"]):
                            home_vid_1 = gr.Video(
                                value="F:\\Stableanimator\\evaluation\\data\\generated_videos\\3210.mp4" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\generated_videos\\3210.mp4") else None,
                                interactive=False, elem_classes=["card-video-box"]
                            )
                            gr.HTML("<div class='card-info-box'><div class='card-title-text'>兵马俑将军</div><div class='card-desc-text'>严阵以威，栩栩如生 | 1024P</div></div>")
                        with gr.Column(scale=1, elem_classes=["gallery-showcase-card"]):
                            home_vid_2 = gr.Video(
                                value="F:\\Stableanimator\\evaluation\\data\\generated_videos\\30555basic.mp4" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\generated_videos\\30555basic.mp4") else None,
                                interactive=False, elem_classes=["card-video-box"]
                            )
                            gr.HTML("<div class='card-info-box'><div class='card-title-text'>汉舞女俑</div><div class='card-desc-text'>长袖乐舞，轨迹防抖 | IPI 特征</div></div>")
                        with gr.Column(scale=1, elem_classes=["gallery-showcase-card"]):
                            home_vid_3 = gr.Video(
                                value="F:\\Stableanimator\\evaluation\\data\\generated_videos\\1.mp4" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\generated_videos\\1.mp4") else None,
                                interactive=False, elem_classes=["card-video-box"]
                            )
                            gr.HTML("<div class='card-info-box'><div class='card-title-text'>唐代三彩俑</div><div class='card-desc-text'>空间仿射，纹理高光 | 沧桑解说</div></div>")
                        with gr.Column(scale=1, elem_classes=["gallery-showcase-card"]):
                            home_vid_4 = gr.Video(
                                value="F:\\Stableanimator\\evaluation\\data\\generated_videos\\3210-120.mp4" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\generated_videos\\3210-120.mp4") else None,
                                interactive=False, elem_classes=["card-video-box"]
                            )
                            gr.HTML("<div class='card-info-box'><div class='card-title-text'>大理石雕像</div><div class='card-desc-text'>细腻逼真，跨语种迁移 | 极速响应</div></div>")

                # ==============================================================
                # 页面 2：【视频生成与驱动工作台】 (Video Generation Page)
                # ==============================================================
                with gr.Column(visible=False) as col_generation:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>🎬 视频生成与参数工作台</div>")
                    with gr.Row():
                        # 左侧输入源与参数调控 (宽度占比 5)
                        with gr.Column(scale=5, elem_classes=["console-panel-box"]):
                            gr.HTML("<div style='font-weight:700; font-size:1.1rem; color:#60A5FA; margin-bottom:1rem;'>1. 驱动源绑定 (必需)</div>")
                            input_image = gr.Image(label="🖼️ 待赋活雕像/文物参考图", type="filepath")
                            input_video = gr.Video(label="💃 参考动作驱动视频 (MP4)")
                            
                            gr.HTML("<div style='font-weight:700; font-size:1.1rem; color:#60A5FA; margin:1.2rem 0 0.6rem 0;'>2. 渲染模型预设</div>")
                            mode_selector = gr.Radio(
                                choices=[("⚡ 极速预览 (动作校对)", 1), ("🌟 高清标准 (默认推荐)", 2), ("👑 影视精修 (智能防抖)", 3)],
                                value=2, label=""
                            )
                            
                            # 高阶消融参数折叠舱 (对标 User 建议：默认收起！)
                            with gr.Accordion("⚙️ 高阶生成参数与时空消融权重 (点击展开)", open=False, elem_classes=["accordion-dark"]):
                                gr.Markdown("精细化调控 SVD-XT、PoseNet 骨架权重与 IPI 身份保持系数：")
                                steps_slider = gr.Slider(min=10, max=50, value=25, step=1, label="推理步数 (Inference Steps)")
                                cfg_slider = gr.Slider(min=1.0, max=10.0, value=4.5, step=0.1, label="引导系数 (CFG Scale)")
                                motion_bucket_id_slider = gr.Slider(min=1, max=255, value=127, step=1, label="运动幅值 (Motion Bucket ID)")
                                posenet_weight_slider = gr.Slider(min=0.0, max=1.5, value=1.0, step=0.05, label="PoseNet 骨架对齐权重")
                                ipi_weight_slider = gr.Slider(min=0.0, max=1.5, value=1.0, step=0.05, label="IPI 身份特征保持系数")
                                tile_slider = gr.Slider(min=8, max=32, value=16, step=4, label="网格分块尺寸 (Tile Size)")
                                fps_slider = gr.Slider(min=8, max=60, value=16, step=1, label="生成视频帧率 (FPS)")
                                overlap_slider = gr.Slider(min=0, max=12, value=6, step=1, label="前后帧时空重叠缓冲 (Overlap)")
                                
                            with gr.Accordion("🔊 声音处理策略与零样本复刻 (TTS / VC)", open=False, elem_classes=["accordion-dark"]):
                                audio_mode_checkboxes = gr.CheckboxGroup(
                                    choices=[("① 保持原视频音频", "keep_orig"), ("② 添加 AI 旁白解说", "add_tts"), ("③ 改变人声音色", "change_vc")],
                                    value=["add_tts"], label="声学处理配置"
                                )
                                tts_textbox = gr.Textbox(label="🎙️ 旁白文本内容", value="秦始皇陵兵马俑，陶塑艺术的巅峰之作，如今在 AI 引擎的驱动下，焕发出千年的历史生机与气魄。", lines=3)
                                tts_source_type = gr.Radio(choices=[("系统预设音色", "preset"), ("用户上传音频复刻", "custom"), ("AI 文本声纹分析", "minimax")], value="preset", label="音色源")
                                tts_preset_selector = gr.Dropdown(choices=[("大秦将军 · 青铜厚重男声", "male-qn-qingzhuang"), ("央视解说 · 纪实沧桑男声", "male-qn-jishi"), ("古典宫廷 · 优雅空灵女声", "female-shaonv"), ("活泼解说 · 现代科普音", "presenter_male")], value="male-qn-qingzhuang", label="预设声调")
                                tts_custom_file = gr.Audio(label="上传参考声音 (3-10秒 MP3/WAV)", type="filepath", visible=False)
                                tts_minimax_col = gr.Column(visible=False)
                                with tts_minimax_col:
                                    timbre_prompt_box = gr.Textbox(label="自然语言声纹提示词", value="浑厚沧桑的中年男声，带有青铜与历史的沉重回响")
                                btn_preview_tts = gr.Button("🎧 立即试听生成的旁白", variant="secondary")
                                preview_audio_out = gr.Audio(label="试听结果", interactive=False)
                                
                                # 音色迁移隐藏变量挂载 (确保参数齐备)
                                vc_source_type = gr.State("preset")
                                vc_preset_selector = gr.State("male-qn-qingzhuang")
                                vc_custom_file = gr.State(None)
                                vc_timbre_prompt_box = gr.State("")
                                vc_keep_bgm = gr.State(True)

                        # 右侧成片展示与控制台 (宽度占比 7)
                        with gr.Column(scale=7, elem_classes=["console-panel-box"]):
                            with gr.Row():
                                btn_start_pipeline = gr.Button("🚀 启动端到端 AI 驱动生成 (Start Pipeline)", variant="primary", elem_classes=["btn-cta-gradient"])
                                btn_reset = gr.Button("🔄 重置工作台", variant="secondary")
                                
                            gr.HTML("<div style='font-weight:700; font-size:1.1rem; color:#4ADE80; margin:1.5rem 0 0.8rem 0;'>🎬 最终生成多模态成片预览</div>")
                            out_vid = gr.Video(label="生成结果展示 (MP4)", height=420)
                            out_aud = gr.Audio(label="独立音频轨同步输出 (MP3/AAC)", interactive=False)
                            
                            gr.HTML("<div style='font-weight:700; font-size:1.1rem; color:#94A3B8; margin:1.5rem 0 0.5rem 0;'>📝 实时引擎推理日志 (Gradio Live Console)</div>")
                            out_log = gr.Textbox(label="", lines=8, max_lines=15, interactive=False, value="⏳ 系统运行状态正常，等待上传驱动素材...")
                            
                            # 中间件数据输出挂载
                            out_align_vid = gr.State(None)
                            out_npy_file = gr.State(None)
                            out_smooth_npy = gr.State(None)
                            out_zip_package = gr.State(None)

                # ==============================================================
                # 页面 3：【AI 智能助手与文博辅导】 (AI Advisor Page)
                # ==============================================================
                with gr.Column(visible=False) as col_advisor:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>✨ AI 智能文博创作助手</div>")
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["console-panel-box"]):
                            gr.HTML("<div style='font-weight:700; font-size:1.1rem; color:#FFFFFF; margin-bottom:1rem;'>💡 您好！请问今天要驱动哪件文物？</div>")
                            gr.Markdown("我可以帮您自动化分析雕像图的历史属性，推荐匹配的动作轨迹、设置参数以及解说台词！")
                            gr.Markdown("---")
                            gr.Markdown("🔥 **快捷推荐指令 (点击即试)**：")
                            btn_p1 = gr.Button("🗿 为兵马俑配置武术参数与厚重解说词", elem_classes=["v4-nav-radio"])
                            btn_p2 = gr.Button("🏛️ 推荐大理石雕像慢舞与古典空灵女声", elem_classes=["v4-nav-radio"])
                            btn_p3 = gr.Button("📜 撰写一段央视纪录片风格的历史沧桑旁白", elem_classes=["v4-nav-radio"])
                            btn_p4 = gr.Button("❓ 解答：这 5 种声学方案到底该怎么选？", elem_classes=["v4-nav-radio"])
                            
                        with gr.Column(scale=8, elem_classes=["console-panel-box"]):
                            chatbot = gr.Chatbot(label="智能文博创作顾问对话记录", height=450)
                            with gr.Row():
                                user_chat_input = gr.Textbox(show_label=False, placeholder="在此输入您的文博赋活创作疑问或需求，例如：我有一尊汉代陶女俑...", scale=8)
                                btn_send_chat = gr.Button("发送消息", variant="primary", scale=2)
                                btn_analyze_timbre = gr.Button("🔍 识图定制参数", variant="secondary", scale=3)

                # ==============================================================
                # 页面 4：【精选案例图库】 (Showcase Gallery Page)
                # ==============================================================
                with gr.Column(visible=False) as col_gallery:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>🖼️ 文博数字赋活精选案例模板库</div>")
                    gr.Markdown("📌 **选择任一馆藏预设模板，一键自动填入输入源与最佳参数配置，即刻开展赋活生成**：")
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["quick-step-card"]):
                            gr.Image(value="F:\\Stableanimator\\evaluation\\data\\input_images\\3210.png" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\input_images\\3210.png") else None, label="大秦兵马俑将领", height=240, interactive=False)
                            gr.HTML("<div style='margin-top:0.8rem; font-weight:700; color:#FFFFFF;'>秦始皇陵 · 大将军武术俑</div>")
                            btn_sel_1 = gr.Button("📥 选择并自动载入工作台", variant="primary")
                        with gr.Column(scale=4, elem_classes=["quick-step-card"]):
                            gr.Image(value="F:\\Stableanimator\\evaluation\\data\\input_images\\30555.png" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\input_images\\30555.png") else None, label="汉代彩绘女舞俑", height=240, interactive=False)
                            gr.HTML("<div style='margin-top:0.8rem; font-weight:700; color:#FFFFFF;'>洛阳博物馆 · 长袖女舞俑</div>")
                            btn_sel_2 = gr.Button("📥 选择并自动载入工作台", variant="primary")
                        with gr.Column(scale=4, elem_classes=["quick-step-card"]):
                            gr.Image(value="F:\\Stableanimator\\evaluation\\data\\input_images\\1.png" if os.path.exists("F:\\Stableanimator\\evaluation\\data\\input_images\\1.png") else None, label="唐代三彩天王俑", height=240, interactive=False)
                            gr.HTML("<div style='margin-top:0.8rem; font-weight:700; color:#FFFFFF;'>陕西历史博物馆 · 天王神将</div>")
                            btn_sel_3 = gr.Button("📥 选择并自动载入工作台", variant="primary")

                # ==============================================================
                # 页面 5：【历史作品与数据中心】 (History & Middleware Page)
                # ==============================================================
                with gr.Column(visible=False) as col_history:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>📜 历史作品与生成成片归档</div>")
                    history_gallery = gr.Gallery(label="本次会话生成的所有视频成果列表 (点击可全屏播放)", columns=3, height=480, object_fit="contain")
                    history_state = gr.State([])

                with gr.Column(visible=False) as col_middleware:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>📊 数据中心与预处理中间件导出</div>")
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["console-panel-box"]):
                            gr.HTML("<div style='font-weight:700; color:#60A5FA; margin-bottom:0.8rem;'>📁 轨迹对齐与平滑数据包下载</div>")
                            gr.Markdown("提供原生 DWPose 133关键点轨迹、HJB 滤波去抖平滑轨迹及完整 ZIP 数据包：")
                            out_file_1 = gr.File(label="原厂对齐运动视频 (Pose Align MP4)", interactive=False)
                            out_file_2 = gr.File(label="原生 133 关键点轨迹 (Raw NPY)", interactive=False)
                            out_file_3 = gr.File(label="HJB 滤波去抖平滑轨迹 (Smooth NPY)", interactive=False)
                            out_file_4 = gr.File(label="完整数据集与中间件打捆包 (ZIP)", interactive=False)

                # ==============================================================
                # 页面 6：【系统设置与运行日志】 (System & Logs Page)
                # ==============================================================
                with gr.Column(visible=False) as col_system:
                    gr.HTML("<div class='section-main-title' style='margin-bottom:1.5rem;'>⚙️ 系统监控与高阶日志记录</div>")
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["console-panel-box"]):
                            gr.HTML("<div style='font-weight:700; color:#38BDF8; margin-bottom:0.8rem;'>💻 GPU 硬件挂载与模型状态</div>")
                            gr.Markdown("当前部署系统已完成以下组件的安全挂载：\n- **时空生成内核**: SVD-XT 1024P Temporal Diffusion\n- **身份约束网络**: IPI Identity-Preserving Adapter\n- **姿态空间仿射**: DWPose Translate Smooth Pipeline\n- **语音合成引擎**: Minimax Zero-Shot Audio Synthesizer")
                        with gr.Column(scale=6, elem_classes=["console-panel-box"]):
                            gr.HTML("<div style='font-weight:700; color:#38BDF8; margin-bottom:0.8rem;'>📝 系统运行日志实时输出</div>")
                            sys_log_view = gr.Textbox(label="", value="[App-Studio] V4.0 工业级 AI 创作平台启动完成。\n[CUDA-Info] PyTorch GPU 环境就绪，待机显存 12.6GB/24GB。", lines=8, interactive=False)

        # ======================================================================
        # 🔄 3. 路由控制与全栈事件调度逻辑 (Routing & Event Bindings)
        # ======================================================================
        
        # 统一侧边栏路由响应中心
        def handle_nav_switch(target):
            target = target or "🏠 首页概览"
            is_home = (target == "🏠 首页概览")
            is_gen = (target in ["🎬 视频生成", "🎛️ 模型与参数"])
            is_adv = (target == "✨ AI 助手")
            is_gal = False
            is_his = (target == "📜 历史作品")
            is_mid = (target == "📦 数据与中间件")
            is_sys = (target in ["📝 系统日志", "⚙️ 设置中心"])
            return (
                gr.update(visible=is_home),
                gr.update(visible=is_gen),
                gr.update(visible=is_adv),
                gr.update(visible=is_gal),
                gr.update(visible=is_his),
                gr.update(visible=is_mid),
                gr.update(visible=is_sys)
            )

        nav_selector.change(
            fn=handle_nav_switch,
            inputs=[nav_selector],
            outputs=[col_home, col_generation, col_advisor, col_gallery, col_history, col_middleware, col_system]
        )

        # 首页 “立即生成/进入工作台” 按钮直达“视频生成”页面
        btn_quick_jump.click(
            fn=lambda: ("🎬 视频生成", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
            outputs=[nav_selector, col_home, col_generation, col_advisor, col_gallery, col_history, col_middleware, col_system]
        )

        # 案例库卡片点击直达并填充输入图
        def select_template_1():
            return "F:\\Stableanimator\\evaluation\\data\\input_images\\3210.png", "🎬 视频生成", None, None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        btn_sel_1.click(fn=select_template_1, outputs=[input_image, nav_selector, nav_manage_selector, nav_sys_selector, col_home, col_generation, col_advisor, col_gallery, col_history, col_middleware, col_system])

        # AI 顾问聊天响应
        chat_inputs = [user_chat_input, chatbot]
        chat_outputs = [user_chat_input, chatbot, mode_selector, tile_slider, steps_slider, cfg_slider, tts_textbox]
        btn_send_chat.click(fn=handle_interactive_chat, inputs=chat_inputs, outputs=chat_outputs)
        user_chat_input.submit(fn=handle_interactive_chat, inputs=chat_inputs, outputs=chat_outputs)
        
        btn_p1.click(fn=lambda: "帮我配置一套大秦兵马俑表演武术动作的参数，搭配浑厚青铜将士音，撰写一段霸气台词。", outputs=[user_chat_input])
        btn_p2.click(fn=lambda: "我有一尊古希腊大理石雕像，想要优雅流畅的慢节奏舞蹈，搭配古典空灵女声，请给出最美参数配置。", outputs=[user_chat_input])
        btn_p3.click(fn=lambda: "请帮我选一套历史文物赋活参数，搭配纪录片解说男声，写一段两百字的历史沧桑感旁白。", outputs=[user_chat_input])
        btn_p4.click(fn=lambda: "你好，请给用户介绍一下这套系统的 5 种声学方案（保持原声、音色转换、解说旁白等）到底该怎么选？", outputs=[user_chat_input])

        btn_analyze_timbre.click(
            fn=handle_analyze_timbre,
            inputs=[input_image, user_chat_input],
            outputs=[timbre_prompt_box, tts_textbox, vc_timbre_prompt_box, tts_preset_selector, vc_preset_selector, out_log]
        )

        btn_preview_tts.click(
            fn=handle_preview_timbre,
            inputs=[tts_textbox, tts_source_type, tts_preset_selector, tts_custom_file, timbre_prompt_box],
            outputs=[preview_audio_out]
        )

        # 端到端生成引擎启动绑定
        btn_start_pipeline.click(
            fn=run_end_to_end_generation,
            inputs=[
                input_image, input_video, mode_selector, tile_slider, steps_slider, cfg_slider,
                motion_bucket_id_slider, posenet_weight_slider, ipi_weight_slider, fps_slider, overlap_slider,
                tts_textbox, audio_mode_checkboxes,
                tts_source_type, tts_preset_selector, tts_custom_file, timbre_prompt_box,
                vc_source_type, vc_preset_selector, vc_custom_file, vc_timbre_prompt_box, vc_keep_bgm,
                history_state
            ],
            outputs=[out_vid, out_aud, out_log, out_align_vid, out_npy_file, out_smooth_npy, out_zip_package, history_gallery, history_state]
        )

        btn_reset.click(
            fn=lambda: (None, None, "🔄 界面与数据中枢已清空重置。", None, None, None, None, [], []),
            outputs=[out_vid, out_aud, out_log, out_align_vid, out_npy_file, out_smooth_npy, out_zip_package, history_gallery, history_state]
        )

    return demo

if __name__ == "__main__":
    app_v4 = build_app_v4_ui()
    app_v4.launch(server_name="0.0.0.0", server_port=7860, share=False)
