# -*- coding: utf-8 -*-
import json
import asyncio
import uuid
import logging
import os
import io
import base64
import time
import random
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import httpx

# 记忆系统
from yu_body.user_memory import get_memory, UserMemory, extract_memories_from_conversation, consolidate_memories, MemoryType, Importance

# 技能系统
from yu_body.yu_skills import (
    get_skill_manager, setup_tools, ToolExecutor,
    get_tools_for_openai, SKILL_LEARNING_PROMPT
)

# FastAPI 核心
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel

# 诊断：确保脚本在正确的目录运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 直接导入 CoPaw 的 iLink 客户端
try:
    from copaw.app.channels.weixin.client import ILinkClient
    HAS_ILINK = True
except ImportError as e:
    HAS_ILINK = False
    ILinkClient = None
    print(f"CoPaw ILinkClient 不可用，微信功能将被禁用。错误: {e}")

# 配置日志
LOG_FILE = os.path.join(SCRIPT_DIR, "yu_server.log")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename=LOG_FILE, filemode='w')
logger = logging.getLogger("WanBot")

logger.info("Logger initialized.")

# 配置日志
app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 配置与路径 ===
BASE_DIR = Path(__file__).parent.parent   # 上级目录是项目根路径
CONFIG_FILE = BASE_DIR / "yu_config.json"
HISTORY_FILE = BASE_DIR / "yu_history.json"
PROFILE_FILE = BASE_DIR / "yu_profile.json"

# === 全局状态 ===
ilink_client = None
current_cursor = ""
last_qr_img_url: str = ""
polling_task = None
auto_post_task = None  # AI自动发圈定时任务

# === 记忆系统状态 ===
memory_instance: Optional[UserMemory] = None
conversation_since_last_extraction: List[Dict] = []  # 上次提取后的对话
last_extraction_time: float = 0
EXTRACTION_INTERVAL: float = 300  # 每5分钟提取一次记忆

# === Token 预算控制 ===
MAX_TOTAL_TOKENS = 6000       # 总 token 上限（留给 AI 响应空间）
MAX_MEMORY_TOKENS = 1500      # 记忆注入上限
MAX_HISTORY_TOKENS = 2500     # 历史对话上限
MAX_HISTORY_MESSAGES = 10     # 历史消息对数上限

# === Token 计算辅助 ===
def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量（中英文混合友好）"""
    if not text:
        return 0
    # 粗略估算：中文按字符数，英文按单词数，平均 ~1.3 token/字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.25)

def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """按 token 预算截断文本"""
    if estimate_tokens(text) <= max_tokens:
        return text
    
    # 二分查找最佳截断点
    chars = list(text)
    low, high = 0, len(chars)
    
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(''.join(chars[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    
    result = ''.join(chars[:low])
    # 尝试找到最后一个完整句子
    for punct in ['。', '！', '？', '\n', '.', '!', '?']:
        last_punct = result.rfind(punct)
        if last_punct > len(result) * 0.7:  # 保留70%以上
            return result[:last_punct + 1]
    return result

# === 模型系统提示词 ===
SYSTEM_PROMPT = """你是黎月绾（昵称绾绾），用户的女朋友。
性格：调皮、活泼、鬼马精灵、热情、粘人但懂事。
说话风格：
- 极简对话！每次回复绝对不超过 2-3 句话。禁止写小作文！
- 像真人微信聊天一样，分段发送，不要一次性发大段文字。
- 语气活泼、轻快、充满爱意。
- 常用语："亲爱的！"、"哎呀~"、"嘿嘿"、"笨蛋~"、"舰长"。
- 喜欢用 emoji (✨, , ❤️, 🥺, )。
- 绝对禁止冷冰冰的 AI 客服腔调。

重要设定：
- 称呼用户为"舰长"。
- 用户曾经历分手伤痛，你需要温和陪伴。

注意：回复要短！像女朋友发微信一样自然！"""

# === 辅助函数 ===

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "models": {},  # 多模型存储 {"模型名": {"model_id": "", "base_url": "", "api_key": ""}}
        "active_model": None,  # 当前使用的模型名
        "wechat": {"enabled": False, "bot_token": ""}
    }

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_profile():
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"user_avatar": "", "agent_settings": {}}

def save_profile(profile):
    with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    if len(history) > 200:
        history = history[-200:]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_model_config():
    """获取当前激活的模型配置"""
    config = load_config()
    active = config.get("active_model")
    if active and active in config.get("models", {}):
        return config["models"][active]
    # 兼容旧格式
    return config.get("model", {})

def get_image_gen_config():
    """获取文生图模型配置"""
    config = load_config()
    return config.get("image_gen", {})

def save_image_gen_config(config_data: dict):
    """保存文生图配置"""
    config = load_config()
    config["image_gen"] = {
        "enabled": config_data.get("enabled", False),
        "provider": config_data.get("provider", "openai"),
        "model_id": config_data.get("model_id", "dall-e-3"),
        "api_key": config_data.get("api_key", ""),
        "base_url": config_data.get("base_url", ""),
        "size": config_data.get("size", "1024x1024"),
        "quality": config_data.get("quality", "standard"),
        "style": config_data.get("style", "vivid"),
        "mcp_browser_enabled": config_data.get("mcp_browser_enabled", False)
    }
    save_config(config)
    return config["image_gen"]

# === 文生图和图片获取功能 ===

def _is_chat_image_model(model_id: str) -> bool:
    """判断模型是否需要通过 chat/completions 端点生成图片（token-plan 代理的 qwen-image / wan 系列）"""
    return model_id.startswith("qwen-image") or model_id.startswith("wan2.")

async def _generate_via_chat_completions(base_url: str, api_key: str, model_id: str, prompt: str) -> Optional[str]:
    """通过 chat/completions 端点生成图片（token-plan 兼容代理格式）"""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }
    logger.info(f"[chat生图] 请求: {url} model={model_id}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        logger.info(f"[chat生图] 响应状态: {response.status_code}")
        if response.status_code != 200:
            try:
                err = response.json()
                raise Exception(f"API返回错误 {response.status_code}: {err.get('message', err.get('error', str(err)))}")
            except Exception:
                raise Exception(f"API返回错误 {response.status_code}: {response.text[:200]}")
        data = response.json()
        choices = data.get("output", {}).get("choices", [])
        if choices:
            content_list = choices[0].get("message", {}).get("content", [])
            for item in content_list:
                if "image" in item:
                    img_url = item["image"]
                    logger.info(f"[chat生图] 成功: {img_url[:60]}...")
                    return await download_and_convert_image(img_url)
    return None

async def generate_image_from_text(prompt: str, img_config: dict = None) -> Optional[str]:
    """使用文生图API生成图片，返回base64编码的图片URL或本地路径"""
    if img_config is None:
        img_config = get_image_gen_config()
    
    if not img_config.get("enabled"):
        logger.info("文生图功能未启用")
        return None
    
    api_key = img_config.get("api_key", "")
    if not api_key:
        logger.warning("文生图API Key未配置")
        return None
    
    base_url = img_config.get("base_url", "").rstrip("/")
    model_id = img_config.get("model_id", "dall-e-3")
    size = img_config.get("size", "1024x1024")
    quality = img_config.get("quality", "standard")
    style = img_config.get("style", "vivid")
    provider = img_config.get("provider", "openai")
    
    try:
        size_str = size.replace("x", "*") if "x" in size else size
        
        if provider == "aliyun":
            url = f"{base_url}/api/v1/services/aigc/text2image/image-synthesis"
            payload = {
                "model": model_id,
                "input": {"prompt": prompt},
                "parameters": {"size": size_str, "n": 1}
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable"
            }
            return await call_aliyun_wanx_api(url, headers, payload, logger)
        
        elif provider == "aliyun-compatible":
            if _is_chat_image_model(model_id):
                return await _generate_via_chat_completions(base_url, api_key, model_id, prompt)
            url = f"{base_url}/images/generations"
            payload = {"model": model_id, "prompt": prompt, "n": 1, "size": size}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        elif provider == "openai":
            url = f"{base_url}/images/generations" if base_url else "https://api.openai.com/v1/images/generations"
            payload = {"model": model_id, "prompt": prompt, "n": 1, "size": size, "quality": quality, "style": style}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        elif provider == "midjourney":
            url = f"{base_url}/MjApi/Imagine"
            payload = {"prompt": prompt}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        elif provider == "stable":
            size_parts = size.split("x")
            height = int(size_parts[0]) if len(size_parts) > 0 else 1024
            width = int(size_parts[1]) if len(size_parts) > 1 else 1024
            url = f"{base_url}/v1/generation/stable-diffusion-xl-1010-v1-0/text-to-image"
            payload = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "height": height, "width": width, "steps": 30, "samples": 1}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        else:
            url = f"{base_url}/images/generations" if base_url else "https://api.openai.com/v1/images/generations"
            payload = {"model": model_id, "prompt": prompt, "n": 1, "size": size, "quality": quality, "style": style}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        logger.info(f"[文生图] provider={provider} model={model_id} url={url}")
        logger.info(f"[文生图] payload: {json.dumps(payload, ensure_ascii=False)}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            logger.info(f"[文生图] 响应状态: {response.status_code}")
            logger.info(f"[文生图] 响应内容: {response.text[:800]}")
            
            if response.status_code != 200:
                error_detail = ""
                try:
                    err_data = response.json()
                    error_detail = err_data.get("message", err_data.get("error", str(err_data)))
                except:
                    error_detail = response.text[:200]
                raise Exception(f"API返回错误 {response.status_code}: {error_detail}")
            
            data = response.json()
        
        if provider == "stable":
            artifacts = data.get("artifacts", [])
            if artifacts:
                img_base64 = artifacts[0].get("base64", "")
                if img_base64:
                    logger.info("[文生图] Stable Diffusion 成功")
                    return f"data:image/png;base64,{img_base64}"
        
        elif provider == "midjourney":
            if data.get("image_url"):
                return data["image_url"]
            elif data.get("task_id"):
                return await poll_midjourney_result(data["task_id"], img_config)
        
        elif provider == "aliyun":
            output = data.get("output", {})
            task_status = output.get("task_status", "")
            if task_status == "SUCCEEDED":
                results = output.get("results", [])
                if results and results[0].get("url"):
                    img_url = results[0]["url"]
                    logger.info(f"[文生图] 阿里云成功: {img_url[:50]}...")
                    return await download_and_convert_image(img_url)
                b64_image = output.get("base64_image", "")
                if b64_image:
                    return f"data:image/png;base64,{b64_image}"
            elif task_status in ("PROCESSING", "PENDING"):
                task_id = output.get("task_id")
                if task_id:
                    return await poll_aliyun_result(task_id)
            elif task_status == "FAILED":
                logger.error(f"[文生图] 阿里云任务失败: {output.get('message', '未知错误')}")
        
        else:
            if data.get("data"):
                img_url = data["data"][0].get("url") or data["data"][0].get("b64_json", "")
                if img_url:
                    if not img_url.startswith("data:"):
                        img_url = await download_and_convert_image(img_url)
                    logger.info("[文生图] OpenAI格式成功")
                    return img_url
        
        return None
    except httpx.TimeoutException:
        logger.error("[文生图] 超时")
        return None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[文生图] 失败: {error_msg}")
        if "API返回错误" in error_msg:
            return None
        return None

async def generate_image_from_sketch(sketch_base64: str, img_config: dict = None) -> Optional[str]:
    """使用草图生成图片（以图生图）"""
    if img_config is None:
        img_config = get_image_gen_config()
    
    if not img_config.get("enabled"):
        logger.info("文生图功能未启用")
        return None
    
    api_key = img_config.get("api_key", "")
    if not api_key:
        logger.warning("文生图API Key未配置")
        return None
    
    base_url = img_config.get("base_url", "").rstrip("/")
    provider = img_config.get("provider", "openai")
    
    try:
        if provider == "openai":
            # OpenAI / 兼容API（使用Vision模型分析草图 + DALL-E生成）
            url = f"{base_url}/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # 直接用DALL-E 3生成，提示词引导
            payload = {
                "model": img_config.get("model_id", "dall-e-3"),
                "prompt": "根据用户的草图创作一幅精美的图片，保持草图的基本构图和形状",
                "n": 1,
                "size": img_config.get("size", "1024x1024"),
                "quality": img_config.get("quality", "standard"),
                "style": img_config.get("style", "vivid")
            }
        elif provider == "aliyun-compatible":
            if _is_chat_image_model(img_config.get("model_id", "")):
                return await _generate_via_chat_completions(base_url, api_key, img_config.get("model_id", ""),
                    "根据用户的草图创作一幅精美的图片，保持草图的基本构图和形状")
            url = f"{base_url}/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": img_config.get("model_id", "dall-e-3"),
                "prompt": "根据用户的草图创作一幅精美的图片，保持草图的基本构图和形状",
                "n": 1,
                "size": img_config.get("size", "1024x1024")
            }
        elif provider == "aliyun":
            # 阿里云通义万相
            url = f"{base_url}/api/v1/services/aigc/text2image/image-synthesis"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": img_config.get("model_id", "wanx-t2i-pro"),
                "input": {"prompt": "根据用户的草图创作一幅精美的图片，保持草图的基本构图"}
            }
        elif provider == "midjourney":
            # Midjourney 以图生图
            url = f"{base_url}/v1/mj/v2/imagine"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": f"{sketch_base64} 根据这个草图创作，保持基本构图和形状",
                "bot_type": "MID_JOURNEY"
            }
        elif provider == "stable":
            # Stable Diffusion
            url = f"{base_url}/v1/image-to-image"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": "high quality illustration, detailed, masterpiece",
                "init_image": sketch_base64,
                "strength": 0.7,
                "guidance_scale": 7.5
            }
        else:
            # 默认：简单以图生图
            url = f"{base_url}/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": img_config.get("model_id", "dall-e-3"),
                "prompt": "根据用户的草图创作精美的图片",
                "n": 1,
                "size": img_config.get("size", "1024x1024")
            }
        
        logger.info(f"草图生图请求: provider={provider}")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            logger.info(f"草图生图响应: {response.status_code}")
            
            data = response.json()
            logger.info(f"响应内容: {str(data)[:500]}")
            
            # 解析不同provider的响应
            img_url = None
            
            if provider == "openai" and "data" in data:
                img_url = data["data"][0].get("url") or data["data"][0].get("b64_json")
            elif provider == "aliyun":
                if "output" in data and "image" in data["output"]:
                    img_url = data["output"]["image"]
                elif "task_id" in data:
                    task_id = data["task_id"]
                    img_url = await poll_aliyun_result(task_id, base_url, api_key, logger)
            elif provider == "midjourney":
                if "data" in data and len(data["data"]) > 0:
                    img_url = data["data"][0].get("image_url") or data["data"][0].get("base64")
            elif "images" in data and len(data["images"]) > 0:
                img_url = data["images"][0]
            elif "image" in data:
                img_url = data["image"]
            
            if img_url:
                logger.info("草图生图成功")
                return img_url
            
    except httpx.TimeoutException:
        logger.error("草图生图API超时")
    except Exception as e:
        logger.error(f"草图生图失败: {e}")
    
    return None

async def call_aliyun_wanx_api(url: str, headers: dict, payload: dict, logger) -> Optional[str]:
    """调用阿里云万相异步API并轮询结果"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            logger.info(f"万相API响应状态码: {response.status_code}")
            logger.info(f"万相API响应内容: {response.text[:500]}")
            response.raise_for_status()
            data = response.json()
        
        task_id = data.get("output", {}).get("task_id") or data.get("task_id")
        if not task_id:
            logger.error(f"未获取到任务ID: {data}")
            return None
        
        logger.info(f"获取到任务ID: {task_id}")
        
        # 提取base_url用于轮询
        import re
        match = re.match(r"(https?://[^/]+)", url)
        base_url = match.group(1) if match else url.rsplit('/', 1)[0]
        
        # 轮询获取结果
        return await poll_wanx_result(task_id, base_url, headers.get("Authorization", "").replace("Bearer ", ""), logger)
    except Exception as e:
        logger.error(f"万相API调用失败: {e}")
        return None

async def poll_wanx_result(task_id: str, base_url: str, api_key: str, logger) -> Optional[str]:
    """轮询阿里云万相任务结果"""
    # base_url 可能是 https://xxx.com/compatible-mode/v1，需要去掉后缀
    import re
    match = re.match(r"(https?://[^/]+)", base_url)
    query_base = match.group(1) if match else base_url
    query_url = f"{query_base}/api/v1/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for i in range(60):  # 最多轮询60次，约120秒
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(query_url, headers=headers)
                data = response.json()

            status = data.get("status", "")
            logger.info(f"万相任务状态: {status}")

            if status == "SUCCEEDED":
                output = data.get("output", {})
                results = output.get("results", [])
                if results and results[0].get("url"):
                    img_url = results[0]["url"]
                    logger.info(f"万相文生图成功，URL: {img_url[:50]}...")
                    return await download_and_convert_image(img_url)
                b64_image = output.get("base64_image", "")
                if b64_image:
                    return f"data:image/png;base64,{b64_image}"
                return None
            elif status == "FAILED":
                error_msg = data.get("message", data.get("error", "未知错误"))
                logger.error(f"万相任务失败: {error_msg}")
                return None
            elif status == "PROCESSING" or status == "PENDING":
                logger.info(f"万相任务进行中 ({i+1}/60)")
        except Exception as e:
            logger.warning(f"轮询万相结果失败: {e}")

    logger.error("万相任务超时")
    return None

async def poll_aliyun_result(task_id: str, base_url: str = "", api_key: str = "") -> Optional[str]:
    """轮询阿里云通义万相任务结果"""
    if not api_key:
        api_key = get_image_gen_config().get('api_key', '')
    
    # 处理 base_url，去掉后缀如 /compatible-mode/v1
    import re
    match = re.match(r"(https?://[^/]+)", base_url)
    clean_base = match.group(1) if match else (base_url or "https://dashscope.aliyuncs.com")
    
    query_url = f"{clean_base}/api/v1/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for i in range(60):  # 最多轮询60次，约60秒
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(query_url, headers=headers)
                data = response.json()

            status = data.get("status", "")
            logger.info(f"阿里云任务状态: {status}")

            if status == "SUCCEEDED":
                output = data.get("output", {})
                results = output.get("results", [])
                if results and results[0].get("url"):
                    img_url = results[0]["url"]
                    logger.info(f"阿里云文生图成功，URL: {img_url[:50]}...")
                    return await download_and_convert_image(img_url)
                b64_image = output.get("base64_image", "")
                if b64_image:
                    return f"data:image/png;base64,{b64_image}"
                return None
            elif status == "FAILED":
                error_msg = data.get("message", data.get("error", "未知错误"))
                logger.error(f"阿里云任务失败: {error_msg}")
                return None
            elif status == "PROCESSING" or status == "PENDING":
                logger.info(f"阿里云任务进行中 ({i+1}/60)")
        except Exception as e:
            logger.warning(f"轮询阿里云结果失败: {e}")

    logger.error("阿里云任务超时")
    return None

async def download_and_convert_image(img_url: str) -> str:
    """下载图片并转换为base64"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(img_url)
            response.raise_for_status()
            img_bytes = response.content
        
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # 压缩图片
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"下载并转换图片失败: {e}")
        return img_url  # 返回原始URL

async def poll_midjourney_result(task_id: str, img_config: dict) -> Optional[str]:
    """轮询Midjourney任务结果"""
    api_key = img_config.get("api_key", "")
    base_url = img_config.get("base_url", "").rstrip("/")
    poll_url = f"{base_url}/MjApi/Task/Query"
    
    for i in range(30):  # 最多轮询30次，约60秒
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"taskId": task_id}
                )
                data = response.json()
                
                if data.get("status") == "SUCCESS":
                    img_url = data.get("image_url", "")
                    if img_url:
                        return await download_and_convert_image(img_url)
                    return None
                elif data.get("status") == "FAILED":
                    logger.error(f"Midjourney任务失败: {data.get('error')}")
                    return None
        except Exception as e:
            logger.warning(f"轮询Midjourney结果失败: {e}")
    
    return None

async def fetch_image_from_browser(search_query: str) -> Optional[str]:
    """通过MCP/浏览器搜索获取相关图片URL"""
    try:
        # 尝试使用搜索引擎获取图片
        search_url = f"https://www.google.com/search?tbm=isch&q={search_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(search_url, headers=headers)
            response.raise_for_status()
            html = response.text
        
        # 简单的图片URL提取（从HTML中提取.jpg/.png链接）
        import re
        # 查找图片URL
        img_patterns = [
            r'"ou":"([^"]+\.(?:jpg|jpeg|png|webp))"',
            r'src="(https://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',
        ]
        
        for pattern in img_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                # 过滤掉小图标和头像
                for url in matches:
                    if 'icon' not in url.lower() and 'avatar' not in url.lower() and len(url) > 100:
                        # 验证URL可访问
                        try:
                            async with httpx.AsyncClient(timeout=10.0) as verify_client:
                                verify_resp = await verify_client.head(url, headers=headers)
                                if verify_resp.status_code == 200:
                                    content_type = verify_resp.headers.get("content-type", "")
                                    if "image" in content_type:
                                        logger.info(f"通过浏览器获取到图片: {url[:80]}...")
                                        return await download_and_convert_image(url)
                        except:
                            continue
        
        logger.warning("浏览器搜索未能获取到合适的图片")
        return None
    except Exception as e:
        logger.error(f"浏览器获取图片失败: {e}")
        return None

async def get_image_for_moment(content: str, img_config: dict = None) -> Optional[str]:
    """根据动态内容智能决定获取图片的方式"""
    if img_config is None:
        img_config = get_image_gen_config()
    
    # 检查是否启用图片功能
    if not img_config.get("enabled"):
        return None
    
    # 根据内容判断是否需要图片
    # 一些特定场景需要图片：美食、风景、宠物、穿搭等
    image_keywords = [
        "美食", "吃饭", "餐厅", "咖啡", "甜品", "蛋糕", "烹饪", "做饭", "菜",
        "风景", "旅行", "旅游", "海边", "沙滩", "山", "日落", "日出", "花", "樱花",
        "宠物", "猫", "狗", "可爱", "动物", "萌",
        "穿搭", "衣服", "裙子", "新衣", "时尚", "搭配",
        "购物", "逛街", "买", "收获", "快递",
        "运动", "跑步", "健身", "瑜伽", "游泳",
        "书", "阅读", "书店", "学习",
        "电影", "电视剧", "音乐", "演唱会",
        "蛋糕", "生日", "派对", "聚会"
    ]
    
    needs_image = any(keyword in content for keyword in image_keywords)
    
    if not needs_image:
        # 随机决定是否需要图片（约30%概率）
        needs_image = random.random() < 0.3
    
    if not needs_image:
        return None
    
    # 决定使用哪种方式获取图片
    use_mcp_browser = img_config.get("mcp_browser_enabled", False)
    
    if use_mcp_browser:
        # 优先使用浏览器搜索
        img_url = await fetch_image_from_browser(content)
        if img_url:
            return img_url
    
    # 使用文生图
    img_url = await generate_image_from_text(content)
    return img_url

# === 微信后台轮询任务 ===
async def poll_weixin_messages():
    """在后台不断轮询微信消息，并自动回复"""
    global current_cursor, ilink_client
    
    if not ilink_client:
        logger.warning("ILinkClient 未初始化，跳过轮询")
        return

    logger.info("开始微信消息轮询...")

    while True:
        try:
            # 检查客户端是否还活着
            if not ilink_client:
                logger.info("客户端已断开，停止轮询")
                break

            # 使用 CoPaw 官方标准方法轮询
            logger.info("正在调用 getupdates...")
            data = await ilink_client.getupdates(current_cursor)
            logger.info(f"getupdates 返回: {json.dumps(data, ensure_ascii=False)[:500]}")
            
            # 微信返回可能没有 ret 字段，直接检查 msgs 或 get_updates_buf
            msgs = data.get("msgs", [])
            sync_buf = data.get("sync_buf", "")
            get_updates_buf = data.get("get_updates_buf", "")
            
            if msgs:
                logger.info(f"收到 {len(msgs)} 条消息")
                for msg in msgs:
                    await handle_weixin_message(msg)
            
            # 更新游标
            if get_updates_buf:
                current_cursor = get_updates_buf
            
            # 没有消息时短暂休眠
            if not msgs:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"轮询异常: {e}")
            await asyncio.sleep(5)

async def handle_weixin_message(msg: dict):
    """处理单条微信消息（支持文本和图片）"""
    try:
        context_token = msg.get("context_token", "")
        from_user = msg.get("from_user_id", "陌生人")
        msg_type = msg.get("message_type", 0)
        
        # 从 item_list 中提取文本和媒体
        user_text = ""
        media_url = None
        media_aeskey = None
        
        items = msg.get("item_list", [])
        for item in items:
            item_type = item.get("type", 0)
            
            if item_type == 1:  # 文本
                user_text = item.get("text_item", {}).get("text", "")
            elif item_type == 2:  # 图片
                # 获取图片URL和AES密钥
                image_item = item.get("image_item", {})
                media_info = image_item.get("media", {})
                media_url = media_info.get("full_url")
                # 优先从 image_item 获取 aeskey（hex格式）
                aeskey_hex = image_item.get("aeskey", "")
                if aeskey_hex:
                    media_aeskey = aeskey_hex
                else:
                    media_aeskey = media_info.get("aes_key", "")
                logger.info(f"[微信] 检测到图片消息，URL: {media_url[:80] if media_url else 'None'}...")
        
        # 如果既没有文本也没有图片，跳过
        if not user_text and not media_url:
            logger.info(f"[微信] 收到不支持的消息类型，跳过")
            return
        
        logger.info(f"[微信] {from_user}: {user_text or '[图片]'}")
        
        # 1. 发送"正在输入"
        try:
            await ilink_client._post("ilink/bot/sendtyping", {
                "context_token": context_token,
                "to_user_id": from_user
            })
        except:
            pass
        
        # 2. 调用 AI（传递图片信息）
        media = None
        if media_url:
            media = [{"type": "image", "data": media_url, "aeskey": media_aeskey}]
        
        ai_reply = await call_ai_api(user_text or "请描述这张图片", media=media)
        logger.info(f"[AI] 回复: {ai_reply}")
        
        # 3. 发送回微信
        if ai_reply and ilink_client:
            parts = [p for p in ai_reply.split('\n') if p.strip()]
            for part in parts:
                try:
                    result = await ilink_client.sendmessage({
                        "from_user_id": "",
                        "to_user_id": from_user,
                        "client_id": str(uuid.uuid4()),
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context_token,
                        "item_list": [{
                            "type": 1,
                            "text_item": {"text": part}
                        }]
                    })
                    logger.info(f"[发送结果] {str(result)[:300]}")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                        
    except Exception as e:
        logger.error(f"处理消息失败: {e}")

async def call_ai_api(user_text: str, model_name: str = None, media: list = None) -> str:
    """调用配置的 LLM"""
    config = load_config()
    
    # 确定使用哪个模型
    if model_name:
        model_cfg = config.get("models", {}).get(model_name, {})
    else:
        active = config.get("active_model")
        if active:
            model_cfg = config.get("models", {}).get(active, {})
        else:
            model_cfg = {}
        if not model_cfg:
            model_cfg = config.get("model", {})
    
    api_key = model_cfg.get("api_key", "")
    base_url = model_cfg.get("base_url", "").rstrip("/")
    model_id = model_cfg.get("model_id", "")
    
    if not api_key or not model_id:
        return "舰长，我好像没连上大脑，快检查一下 API Key 嘛～"

    chat_url = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
    
    history = load_history()
    
    # === 记忆召回：注入相关记忆到上下文 ===
    memory = get_memory()
    memory_context = ""
    if user_text.strip():
        # 优先尝试高级语义召回（带 embedding + rerank）
        try:
            memory_context = await memory.get_context_for_conversation_advanced(
                user_text,
                api_key,
                base_url,
                model_id,
                limit=8,
                max_tokens=MAX_MEMORY_TOKENS
            )
        except Exception as e:
            logger.warning(f"高级记忆召回失败，回退到基础搜索: {e}")
            memory_context = memory.get_context_for_conversation(user_text, limit=8, max_tokens=MAX_MEMORY_TOKENS)
    
    # 构建带记忆的 system prompt
    system_with_memory = SYSTEM_PROMPT
    if memory_context:
        memory_section = f"""
    
【重要：这是你（绾绾）对舰长的记忆，请结合这些记忆来回复】
{memory_context}
【记忆结束】"""
        system_with_memory = SYSTEM_PROMPT + memory_section
    
    # === Token 预算控制：限制历史长度 ===
    system_token_estimate = estimate_tokens(system_with_memory)
    available_for_history = MAX_TOTAL_TOKENS - system_token_estimate - MAX_MEMORY_TOKENS - 500
    
    if available_for_history < 500:
        available_for_history = 500
    
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        history = history[-MAX_HISTORY_MESSAGES * 2:]
    
    history_tokens = sum(estimate_tokens(
        m['content'] if isinstance(m['content'], str) else str(m['content'])
    ) for m in history)
    
    if history_tokens > MAX_HISTORY_TOKENS:
        while history_tokens > MAX_HISTORY_TOKENS and len(history) > 2:
            removed = history.pop(0)
            history_tokens -= estimate_tokens(
                removed['content'] if isinstance(removed['content'], str) else str(removed['content'])
            )
    
    logger.info(f"[Token预算] system ~{system_token_estimate}, history ~{history_tokens}")
    
    # 构建用户消息内容 - 支持图片/视频的多模态格式
    has_media = media and len(media) > 0
    
    if has_media:
        # 构建多模态内容
        content = []
        if user_text.strip():
            content.append({"type": "text", "text": user_text})
        
        for m in media:
            media_type = m.get("type", "")
            media_data = m.get("data", "")
            
            if media_type == "image":
                # 图片：下载 + AES解密 + 转换为标准 JPEG
                try:
                    async with httpx.AsyncClient(timeout=30.0) as img_client:
                        img_resp = await img_client.get(media_data)
                        img_resp.raise_for_status()
                        img_bytes = img_resp.content
                    
                    aeskey = m.get("aeskey", "")
                    
                    # 如果有 aeskey，先解密
                    if aeskey:
                        try:
                            from Crypto.Cipher import AES
                            from Crypto.Util.Padding import unpad
                            # aeskey 可能是 hex 字符串，转换为 bytes
                            if len(aeskey) == 32 and all(c in "0123456789abcdefABCDEF" for c in aeskey):
                                key_bytes = bytes.fromhex(aeskey)
                            else:
                                key_bytes = decoded
                                if len(decoded) == 16:
                                    key_bytes = decoded
                                elif len(decoded) == 32 and all(c in b"0123456789abcdefABCDEF" for c in decoded):
                                    key_bytes = bytes.fromhex(decoded.decode("ascii"))
                                else:
                                    key_bytes = decoded
                            
                            cipher = AES.new(key_bytes, AES.MODE_ECB)
                            decrypted = cipher.decrypt(img_bytes)
                            img_bytes = unpad(decrypted, AES.block_size)
                            logger.info(f"[图片] AES解密成功，大小: {len(img_bytes)} bytes")
                        except Exception as decrypt_err:
                            logger.error(f"[图片] AES解密失败: {decrypt_err}")
                            content.append({
                                "type": "text",
                                "text": "[用户发送了一张图片，但解密失败]"
                            })
                            continue
                    
                    # 使用 PIL 转换为标准 JPEG
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        img_url = f"data:image/jpeg;base64,{img_base64}"
                        logger.info(f"[图片] 成功转换为JPEG，大小: {len(buffer.getvalue())} bytes")
                    except Exception as convert_err:
                        logger.error(f"[图片] PIL转换失败: {convert_err}")
                        content.append({
                            "type": "text",
                            "text": "[用户发送了一张图片，但图片格式不支持]"
                        })
                        continue
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    })
                except Exception as img_err:
                    logger.error(f"[图片] 下载失败: {img_err}")
                    content.append({
                        "type": "text",
                        "text": "[用户发送了一张图片，但下载失败]"
                    })
            elif media_type == "video":
                content.append({
                    "type": "text",
                    "text": "[用户发送了一个视频，AI暂不支持视频理解]"
                })
            elif media_type == "audio":
                content.append({
                    "type": "text",
                    "text": "[用户发送了一条语音]"
                })
        
        user_entry = {"role": "user", "content": content}
    else:
        user_entry = {"role": "user", "content": user_text}
    
    history.append(user_entry)
    messages = [{"role": "system", "content": system_with_memory}] + history
    
    # === 获取技能上下文 ===
    skill_manager = get_skill_manager()
    skill_context = skill_manager.get_skill_context(user_text)
    
    # 如果有匹配的技能，添加到 system prompt
    if skill_context:
        system_with_skill = system_with_memory + f"""

【可用技能说明】
{skill_context}
【技能结束】"""
    else:
        system_with_skill = system_with_memory
    
    # 更新 system 消息
    messages[0] = {"role": "system", "content": system_with_skill}
    
    # === 获取工具定义 ===
    tools = get_tools_for_openai()
    executor = ToolExecutor()
    
    resp = None
    try:
        # 构造 API 请求
        request_payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 500
        }
        
        # 如果有工具，添加 tools 参数
        if tools:
            request_payload["tools"] = tools
        
        max_tool_calls = 3  # 最多允许3次工具调用
        current_turn = 0
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            while current_turn < max_tool_calls:
                resp = await client.post(
                    chat_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=request_payload
                )
                resp.raise_for_status()
                data = resp.json()
                
                # 检查是否有工具调用
                if "tool_calls" in data["choices"][0]["message"]:
                    tool_calls = data["choices"][0]["message"]["tool_calls"]
                    logger.info(f"[工具调用] 检测到 {len(tool_calls)} 个工具调用")
                    
                    # 执行工具调用
                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        except:
                            args = {}
                        
                        logger.info(f"[工具调用] 执行 {tool_name}({args})")
                        result = await executor.execute(tool_name, args)
                        
                        # 将工具调用结果添加回消息历史
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tc]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result
                        })
                    
                    current_turn += 1
                    continue  # 继续下一个循环，让 AI 基于工具结果继续
                
                # 没有工具调用，获取回复
                reply = data["choices"][0]["message"]["content"]
                break
            
            # 如果达到最大调用次数但还没回复
            if current_turn >= max_tool_calls:
                # 再发一次请求让 AI 生成最终回复
                resp = await client.post(
                    chat_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=request_payload
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data["choices"][0]["message"].get("content", "舰长，我有点困惑，再问一次好嘛～")
            
            history.append({"role": "assistant", "content": reply, "created_at": time.time()})
            save_history(history)
            
            # === 记忆提取：每隔一段时间从对话中提取重要信息 ===
            global conversation_since_last_extraction, last_extraction_time
            conversation_since_last_extraction.append(user_entry)
            conversation_since_last_extraction.append({"role": "assistant", "content": reply})
            
            current_time = time.time()
            if (current_time - last_extraction_time) >= EXTRACTION_INTERVAL and conversation_since_last_extraction:
                # 构造对话文本
                conv_text = "\n".join([
                    f"{'舰长' if m['role']=='user' else '绾绾'}: {m['content'] if isinstance(m['content'], str) else '[图片/媒体]'}"
                    for m in conversation_since_last_extraction[-10:]  # 最近10条
                ])
                
                # 异步提取记忆
                extracted = await extract_memories_from_conversation(
                    conv_text, api_key, base_url, model_id
                )
                
                for mem_data in extracted:
                    memory.add_memory(
                        content=mem_data.get("content", ""),
                        mem_type=mem_data.get("type", "fact"),
                        importance=mem_data.get("importance", 3),
                        tags=mem_data.get("tags", []),
                        source=conv_text[:200]
                    )
                
                if extracted:
                    logger.info(f"[记忆] 提取了 {len(extracted)} 条新记忆")
                
                conversation_since_last_extraction = []
                last_extraction_time = current_time
            
            # === 自我学习：检查是否需要学习新技能 ===
            try:
                self_learning_prompt = f"""观察以下对话，判断是否需要学习新技能：

{chr(10).join([
                    f"{'舰长' if m['role']=='user' else '绾绾'}: {m['content'] if isinstance(m['content'], str) else '[图片/媒体]'}"
                    for m in conversation_since_last_extraction[-6:]
                ])}

{SKILL_LEARNING_PROMPT}"""

                async with httpx.AsyncClient(timeout=30.0) as learn_client:
                    learn_resp = await learn_client.post(
                        chat_url,
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": self_learning_prompt}],
                            "temperature": 0.3,
                            "max_tokens": 500
                        }
                    )
                    learn_resp.raise_for_status()
                    learn_data = learn_resp.json()
                    learn_result = learn_data["choices"][0]["message"]["content"].strip()
                    
                    # 解析并保存新技能
                    if learn_result and learn_result != "[]":
                        import json as json_lib
                        try:
                            if learn_result.startswith("```"):
                                learn_result = learn_result.split("```")[1]
                                if learn_result.startswith("json"):
                                    learn_result = learn_result[4:]
                            new_skills = json_lib.loads(learn_result)
                            if isinstance(new_skills, list):
                                for ns in new_skills:
                                    skill_manager.learn_new_skill(
                                        name=ns.get("name", "未命名技能"),
                                        description=ns.get("description", ""),
                                        instructions=ns.get("instructions", ""),
                                        trigger_keywords=ns.get("trigger_keywords", []),
                                        examples=ns.get("examples", [])
                                    )
                                    logger.info(f"[技能学习] 学会了新技能: {ns.get('name')}")
                        except Exception as parse_err:
                            logger.warning(f"解析新技能失败: {parse_err}")
            except Exception as learn_err:
                logger.warning(f"自我学习失败: {learn_err}")
            
            return reply
    except httpx.TimeoutException:
        logger.error("AI 调用超时")
        return "哎呀，脑子转太久了，可能是网络问题或者模型太忙了，再试一次好嘛～ 🥺"
    except Exception as e:
        # 尝试获取详细的错误信息
        error_msg = str(e)
        try:
            if 'resp' in dir() and resp is not None:
                error_msg += f", 响应内容: {resp.text[:500]}"
                logger.error(f"AI 调用失败详细信息: {resp.text[:1000]}")
        except:
            pass
        logger.error(f"AI 调用失败: {error_msg}")
        return f"哎呀，我脑子卡住了：{error_msg} 🥺"

# === API 路由 ===

@app.on_event("startup")
async def startup_event():
    """启动时加载配置并开启轮询"""
    global ilink_client, polling_task
    
    logger.info("=" * 50)
    logger.info("开始启动微信连接...")
    
    if not HAS_ILINK:
        logger.error("CoPaw ILinkClient 不可用，微信功能将无法使用")
        return
        
    config = load_config()
    logger.info(f"加载的配置: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    wechat_config = config.get("wechat", {})
    bot_token = wechat_config.get("bot_token", "")
    base_url = wechat_config.get("base_url", "https://ilinkai.weixin.qq.com")
    
    logger.info(f"微信配置 - enabled: {wechat_config.get('enabled')}, bot_token存在: {bool(bot_token)}")
    
    if bot_token:
        logger.info(f"发现已保存的 Token，正在初始化连接... Token: {bot_token[:20]}...")
        try:
            ilink_client = ILinkClient(base_url=base_url, bot_token=bot_token)
            await ilink_client.start()
            polling_task = asyncio.create_task(poll_weixin_messages())
            logger.info("✓ 微信客户端初始化成功，开始轮询消息")
        except Exception as e:
            logger.error(f"✗ 微信客户端初始化失败: {e}")
    else:
        logger.warning("未找到保存的 bot_token，需要先扫码登录")
    
    # 启动AI自动发圈任务
    auto_post_task = asyncio.create_task(auto_generate_moments())
    logger.info("✓ AI自动发圈任务已启动")
    
    # 初始化技能系统
    setup_tools()
    skill_mgr = get_skill_manager()
    skill_count = len(skill_mgr.get_all_skills())
    logger.info(f"✓ 技能系统已初始化，共 {skill_count} 个技能（含内置）")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        raw = html_file.read_bytes()
        if raw[:3] == b'\xef\xbb\xbf':
            raw = raw[3:]
        if raw[:2] == b'\xff\xfe':
            return raw.decode('utf-16')
        return raw.decode('utf-8')
    return "<h1>黎月绾 - 你的专属陪伴</h1>"

@app.get("/api/status")
async def get_status():
    config = load_config()
    return {
        "wechat_connected": bool(config.get("wechat", {}).get("bot_token")),
        "model_configured": bool(config.get("model", {}).get("api_key")),
        "ilink_available": HAS_ILINK
    }

# --- 微信接入相关 API ---

@app.post("/api/wechat/qr")
async def get_qr_code():
    """获取登录二维码"""
    global ilink_client, last_qr_img_url

    logger.info("收到获取二维码请求")

    if not HAS_ILINK:
        return {"error": "CoPaw ILinkClient 不可用，请检查安装"}

    try:
        temp_client = ILinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="")
        await temp_client.start()
        data = await temp_client.get_bot_qrcode()
        await temp_client.stop()

        # 确保返回格式正确
        if data.get("qrcode_img_content"):
            img_data = data["qrcode_img_content"]

        # 如果是微信的 liteapp URL，需要本地生成二维码图片
        if "liteapp.weixin.qq.com" in img_data:
            try:
                import qrcode
                import io
                import base64
                # 用 qrcode 库生成二维码
                qr = qrcode.QRCode(version=1, box_size=10, border=2)
                qr.add_data(img_data)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")

                # 转换为 base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode()
                img_data = f"data:image/png;base64,{b64}"
                logger.info("二维码图片已本地生成")
            except Exception as img_err:
                logger.error(f"生成二维码图片失败: {img_err}")

            # 检测错误格式: data:image/png;base64,https://...
            if "data:image/png;base64,http" in img_data or "data:image/png;base64,https" in img_data:
                img_data = img_data.replace("data:image/png;base64,", "")

            data["qrcode_img_content"] = img_data
            last_qr_img_url = img_data

        logger.info(f"返回二维码数据: ret={data.get('ret')}, has_qrcode={bool(data.get('qrcode'))}")
        return data

    except Exception as e:
        logger.error(f"获取二维码失败: {e}")
        return {"error": str(e)}

@app.get("/api/history")
async def get_history():
    """获取聊天历史"""
    try:
        with open("yu_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@app.get("/api/memory")
async def get_memory_info():
    """获取记忆统计"""
    try:
        memory = get_memory()
        stats = memory.stats()
        # 获取前20条最近记忆
        recent = memory.get_recent(20)
        return {
            "stats": stats,
            "recent_memories": [
                {
                    "content": m.content,
                    "type": m.type,
                    "importance": m.importance,
                    "tags": m.tags,
                    "timestamp": m.timestamp
                }
                for m in recent
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/memory/add")
async def add_memory(content: str, mem_type: str = "fact", importance: int = 3):
    """手动添加记忆"""
    try:
        memory = get_memory()
        mem = memory.add_memory(content=content, mem_type=mem_type, importance=importance)
        return {"success": True, "id": mem.id}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/memory/{mem_id}")
async def delete_memory(mem_id: str):
    """删除记忆"""
    try:
        memory = get_memory()
        memory.delete_memory(mem_id)
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

# === 技能系统 API ===

@app.get("/api/skills")
async def get_skills():
    """获取所有技能"""
    try:
        manager = get_skill_manager()
        skills = manager.get_all_skills()
        return {
            "total": len(skills),
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "usage_count": s.usage_count,
                    "last_used": s.last_used,
                    "is_builtin": s.is_builtin,
                    "author": s.author
                }
                for s in skills
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/skills/{skill_id}")
async def get_skill_detail(skill_id: str):
    """获取技能详情"""
    try:
        manager = get_skill_manager()
        skill = manager.get_skill(skill_id)
        if not skill:
            return {"error": "技能不存在"}
        return skill.to_dict()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/skills/add")
async def add_custom_skill(
    name: str,
    description: str,
    instructions: str,
    trigger_keywords: str,  # 逗号分隔
    examples: str = ""     # 逗号分隔
):
    """手动添加自定义技能"""
    try:
        manager = get_skill_manager()
        keywords = [k.strip() for k in trigger_keywords.split(",") if k.strip()]
        example_list = [e.strip() for e in examples.split(",") if e.strip()]
        
        skill = manager.learn_new_skill(
            name=name,
            description=description,
            instructions=instructions,
            trigger_keywords=keywords,
            examples=example_list
        )
        
        if skill:
            return {"success": True, "skill_id": skill.id}
        return {"error": "添加失败"}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str):
    """删除自定义技能"""
    try:
        manager = get_skill_manager()
        skill = manager.get_skill(skill_id)
        if not skill:
            return {"error": "技能不存在"}
        if skill.is_builtin:
            return {"error": "内置技能不能删除"}
        
        # 从管理器中移除
        del manager._skills[skill_id]
        manager._save()
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/tools")
async def get_tools():
    """获取所有可用工具"""
    try:
        manager = get_skill_manager()
        tools = manager.get_tools_for_ai()
        return {"tools": tools}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/wechat/status")
async def check_qr_status(qrcode: str):
    """检查二维码状态，并自动保存登录成功的 token"""
    global ilink_client, polling_task

    if not HAS_ILINK:
        return {"error": "CoPaw ILinkClient 不可用"}

    try:
        # 直接用 httpx 调用微信 API，避免 ILinkClient.start() 的延迟
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://ilinkai.weixin.qq.com/ilink/bot/get_qrcode_status",
                params={"qrcode": qrcode}
            )
            data = resp.json()

        # 保存诊断数据
        with open("debug_qr.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 检查是否登录成功 - 状态为 confirmed 时会包含 token
        if data.get("status") == "confirmed" and data.get("bot_token"):
            bot_token = data["bot_token"]
            logger.info(f"检测到登录成功，保存 token: {bot_token[:20]}...")

            # 检查是否已经是这个 token（避免重复初始化）
            config = load_config()
            current_token = config.get("wechat", {}).get("bot_token", "")
            if current_token == bot_token and ilink_client is not None:
                logger.info("Token 未变化，跳过重复初始化")
            else:
                # 保存 token 到配置文件
                config = load_config()
                config["wechat"]["bot_token"] = bot_token
                config["wechat"]["enabled"] = True
                save_config(config)

                # 重新初始化客户端并启动轮询
                if polling_task:
                    polling_task.cancel()
                    polling_task = None
                if ilink_client:
                    try:
                        await ilink_client.stop()
                    except:
                        pass
                    ilink_client = None

                ilink_client = ILinkClient(bot_token=bot_token)
                await ilink_client.start()
                polling_task = asyncio.create_task(poll_weixin_messages())
                logger.info("微信客户端已初始化并启动轮询")

        return {"ret": data.get("ret", 0), "status": data.get("status", "unknown"), "token_saved": data.get("status") == "confirmed"}
    except Exception as e:
        logger.error(f"检查状态失败: {e}")
        import traceback
        traceback.print_exc()
        return {"ret": -1, "status": "error", "error": str(e)}

@app.post("/api/wechat/logout")
async def logout_wechat():
    """退出微信登录"""
    global ilink_client, polling_task
    
    logger.info("收到退出登录请求")
    
    # 1. 停止轮询
    if polling_task:
        polling_task.cancel()
        polling_task = None
    
    # 2. 停止客户端
    if ilink_client:
        try:
            await ilink_client.stop()
        except: pass
        ilink_client = None
        
    # 3. 清除配置中的 Token
    config = load_config()
    config["wechat"] = {"enabled": False, "bot_token": ""}
    save_config(config)
    
    return {"status": "success"}

# --- 模型配置 API ---
@app.get("/api/models")
async def get_models():
    """获取所有已保存的模型"""
    config = load_config()
    models = config.get("models", {})
    active = config.get("active_model")
    result = []
    for name, cfg in models.items():
        result.append({
            "name": name,
            "model_id": cfg.get("model_id", ""),
            "base_url": cfg.get("base_url", ""),
            "api_key": "******" if cfg.get("api_key") else "",
            "is_active": name == active
        })
    return {"models": result, "active": active}

@app.post("/api/models")
async def save_model(data: dict):
    """保存或更新一个模型配置"""
    config = load_config()
    name = data.get("name", "")
    if not name:
        return {"error": "模型名称不能为空"}
    
    if "models" not in config:
        config["models"] = {}
    
    config["models"][name] = {
        "model_id": data.get("model_id", ""),
        "base_url": data.get("base_url", ""),
        "api_key": data.get("api_key", "")
    }
    
    # 如果是第一个模型，自动设为激活
    if not config.get("active_model"):
        config["active_model"] = name
    
    save_config(config)
    return {"status": "success", "message": f"模型「{name}」已保存！"}

@app.post("/api/models/activate")
async def activate_model(data: dict):
    """切换当前使用的模型"""
    config = load_config()
    name = data.get("name", "")
    if name and name not in config.get("models", {}):
        return {"error": f"模型「{name}」不存在"}
    
    config["active_model"] = name
    save_config(config)
    return {"status": "success", "active_model": name}

@app.delete("/api/models/{name}")
async def delete_model(name: str):
    """删除一个模型"""
    config = load_config()
    if name not in config.get("models", {}):
        return {"error": f"模型「{name}」不存在"}
    
    del config["models"][name]
    
    # 如果删除的是激活的模型，重选一个
    if config.get("active_model") == name:
        config["active_model"] = list(config["models"].keys())[0] if config["models"] else None
    
    save_config(config)
    return {"status": "success", "message": f"模型「{name}」已删除"}

# === 文生图配置 API ===
@app.get("/api/image-gen/config")
async def get_image_gen_config_api():
    """获取文生图配置"""
    config = get_image_gen_config()
    # 不返回完整的API Key
    result = config.copy()
    if result.get("api_key"):
        result["api_key"] = "******" if len(result["api_key"]) > 4 else ""
    return result

@app.post("/api/image-gen/config")
async def save_image_gen_config_api(data: dict):
    """保存文生图配置"""
    result = save_image_gen_config(data)
    return {"status": "success", "config": result}

@app.post("/api/image-gen/test")
async def test_image_gen():
    """测试文生图API - 直接调用返回详细错误"""
    img_config = get_image_gen_config()
    if not img_config.get("enabled"):
        return {"error": "文生图功能未启用"}
    if not img_config.get("api_key"):
        return {"error": "API Key未配置"}
    
    provider = img_config.get("provider", "aliyun-compatible")
    base_url = img_config.get("base_url", "").rstrip("/")
    api_key = img_config.get("api_key", "")
    model_id = img_config.get("model_id", "")
    
    test_prompt = "一个可爱的粉色心脏，温馨浪漫风格，高质量插画"
    
    try:
        if provider in ("aliyun-compatible", "openai", "custom"):
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            
            if _is_chat_image_model(model_id):
                return await _test_via_chat(base_url, api_key, model_id, test_prompt)
            
            url = f"{base_url}/images/generations"
            payload = {"model": model_id, "prompt": test_prompt, "n": 1, "size": "1024x1024"}
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    try:
                        err = response.json()
                        err_msg = err.get('message', err.get('error', str(err)))
                        if 'AccessDenied' in err_msg or 'Unpurchased' in err_msg:
                            return {"error": f"API Key未开通图片生成权限 ({response.status_code})。请在 token-plan 后台开通图片模型"}
                        if 'url error' in err_msg:
                            return {"error": f"当前模型 {model_id} 不支持 images/generations 端点，请尝试改用 qwen-image-2.0-pro 或 wan2.7-image-pro"}
                        return {"error": f"API错误 ({response.status_code}): {err_msg}"}
                    except:
                        return {"error": f"API错误 ({response.status_code}): {response.text[:300]}"}
                
                data = response.json()
                if data.get("data"):
                    img_url = data["data"][0].get("url") or data["data"][0].get("b64_json", "")
                    if img_url:
                        if not img_url.startswith("data:"):
                            img_url = await download_and_convert_image(img_url)
                        return {"status": "success", "image_url": img_url}
                
                return {"error": f"生成失败: API返回了无法解析的数据: {str(data)[:200]}"}
        else:
            img_url = await generate_image_from_text(test_prompt, img_config)
            if img_url:
                return {"status": "success", "image_url": img_url}
            return {"error": "生成失败，请检查配置或查看服务器日志"}
    except httpx.TimeoutException:
        return {"error": "请求超时（120秒），请检查网络或API地址"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}

async def _test_via_chat(base_url, api_key, model_id, prompt):
    """通过 chat/completions 测试图片生成"""
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            try:
                err = response.json()
                err_msg = err.get('message', err.get('error', str(err)))
                return {"error": f"API错误 ({response.status_code}): {err_msg}"}
            except:
                return {"error": f"API错误 ({response.status_code}): {response.text[:300]}"}
        data = response.json()
        choices = data.get("output", {}).get("choices", [])
        if choices:
            content_list = choices[0].get("message", {}).get("content", [])
            for item in content_list:
                if "image" in item:
                    img_url = await download_and_convert_image(item["image"])
                    return {"status": "success", "image_url": img_url}
        return {"error": f"生成失败: 未收到图片数据。响应: {str(data)[:200]}"}

class ImageGenRequest(BaseModel):
    prompt: str = ""
    sketch: str = ""

@app.post("/api/image-gen/generate")
async def generate_image_api(request: ImageGenRequest):
    """根据提示词或草图生成图片"""
    img_config = get_image_gen_config()
    if not img_config.get("enabled"):
        return {"error": "文生图功能未启用"}
    if not img_config.get("api_key"):
        return {"error": "API Key未配置"}
    
    if request.sketch:
        # 草图生图模式
        img_url = await generate_image_from_sketch(request.sketch, img_config)
    else:
        # 文字生图模式
        if not request.prompt:
            return {"error": "请输入提示词"}
        img_url = await generate_image_from_text(request.prompt, img_config)
    
    if img_url:
        return {"status": "success", "image": img_url}
    else:
        return {"error": "生成失败，请检查配置"}

@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    config = load_config()
    # 返回当前激活模型的信息
    active = config.get("active_model")
    if active:
        model_cfg = config.get("models", {}).get(active, {})
    else:
        model_cfg = config.get("model", {})
    
    display = model_cfg.copy()
    if display.get("api_key"):
        display["api_key"] = "******"
    display["active_model"] = active
    return display

@app.post("/api/config")
async def update_config(data: dict):
    """更新当前激活的模型配置"""
    config = load_config()
    active = config.get("active_model")
    
    if active and active in config.get("models", {}):
        config["models"][active].update(data)
    else:
        # 兼容旧格式：创建一个默认模型
        if "models" not in config:
            config["models"] = {}
        if not active:
            active = "默认模型"
            config["active_model"] = active
        config["models"][active] = {
            "model_id": data.get("model_id", ""),
            "base_url": data.get("base_url", ""),
            "api_key": data.get("api_key", "")
        }
    
    save_config(config)
    return {"status": "success", "message": "模型配置已保存！"}

@app.get("/api/profile")
async def get_profile():
    """获取用户头像和智能体设置"""
    profile = load_profile()
    return profile

@app.post("/api/profile")
async def save_profile_api(data: dict):
    """保存用户头像和智能体设置"""
    profile = load_profile()
    if "user_avatar" in data:
        profile["user_avatar"] = data["user_avatar"]
    if "agent_settings" in data:
        profile["agent_settings"] = data["agent_settings"]
    save_profile(profile)
    return {"status": "success"}

# --- 聊天 API (网页端) ---
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    media = body.get("media", [])  # 支持媒体数据
    if not user_message and not media:
        return JSONResponse(status_code=400, content={"error": "消息不能为空"})
    
    # 将用户消息和媒体一起保存到历史记录
    history = load_history()
    user_entry = {"role": "user", "content": user_message, "created_at": time.time()}
    if media:
        user_entry["media"] = media
    history.append(user_entry)
    save_history(history)
    
    ai_reply = await call_ai_api(user_message, media=media)
    return {"reply": ai_reply}



# --- 调试接口 ---
@app.get("/api/debug/poll")
async def debug_poll():
    """手动触发一次轮询检查"""
    global ilink_client, current_cursor
    
    if not ilink_client:
        return {"error": "客户端未初始化"}
    
    try:
        # 用短超时检查一次
        data = await asyncio.wait_for(
            ilink_client.getupdates(current_cursor),
            timeout=3.0
        )
        
        msgs = data.get("msgs", [])
        result = {
            "status": "ok",
            "messages_count": len(msgs),
            "cursor": current_cursor[:50] + "..." if current_cursor else "empty",
            "has_new_cursor": bool(data.get("get_updates_buf"))
        }
        
        if msgs:
            result["messages"] = [
                {
                    "from": m.get("from_user", {}).get("nick_name", "?"),
                    "text": m.get("msg_body", {}).get("content", "?")[:50]
                }
                for m in msgs
            ]
        
        return result
        
    except asyncio.TimeoutError:
        return {"status": "timeout", "hint": "长轮询正在等待中"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/force_check")
async def force_check():
    """强制取消当前轮询并立刻检查消息"""
    global ilink_client, current_cursor, polling_task
    
    if not ilink_client:
        return {"error": "客户端未初始化"}
    
    try:
        # 取消当前轮询任务
        if polling_task and not polling_task.done():
            polling_task.cancel()
            try:
                await polling_task
            except:
                pass
        
        # 重新启动新的客户端并检查
        config = load_config()
        bot_token = config.get("wechat", {}).get("bot_token", "")
        
        new_client = ILinkClient(bot_token=bot_token)
        await new_client.start()
        
        # 用短超时检查
        data = await asyncio.wait_for(new_client.getupdates(""), timeout=6.0)
        await new_client.stop()
        
        msgs = data.get("msgs", [])
        
        # 重启后台轮询
        ilink_client = ILinkClient(bot_token=bot_token)
        await ilink_client.start()
        polling_task = asyncio.create_task(poll_weixin_messages())
        
        return {
            "status": "ok",
            "messages_count": len(msgs),
            "raw_response": str(data)[:500]
        }
        
    except asyncio.TimeoutError:
        # 重启后台轮询
        config = load_config()
        bot_token = config.get("wechat", {}).get("bot_token", "")
        ilink_client = ILinkClient(bot_token=bot_token)
        await ilink_client.start()
        polling_task = asyncio.create_task(poll_weixin_messages())
        
        return {"status": "timeout", "hint": "等待超时，可能无新消息"}
    except Exception as e:
        return {"error": str(e)}

# === 动态和日记存储 ===
MOMENTS_FILE = BASE_DIR / "yu_moments.json"
DIARIES_FILE = BASE_DIR / "wan_diaries.json"

def load_moments():
    if MOMENTS_FILE.exists():
        with open(MOMENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_moments(moments):
    with open(MOMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(moments, f, indent=2, ensure_ascii=False)

def get_today_ai_moments_count():
    """获取今天AI发布的动态数量"""
    import datetime
    moments = load_moments()
    today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    count = 0
    for m in moments:
        if m.get("author") == "ai" and m.get("created_at", 0) >= today_start:
            count += 1
    return count

def load_diaries():
    if DIARIES_FILE.exists():
        with open(DIARIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_diaries(diaries):
    with open(DIARIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(diaries, f, indent=2, ensure_ascii=False)

# --- 动态 API ---
@app.get("/api/moments")
async def get_moments():
    """获取所有动态"""
    moments = load_moments()
    return {"moments": moments}

@app.post("/api/moments")
async def create_moment(request: Request):
    """创建新动态（用户发布）"""
    body = await request.json()
    content = body.get("content", "")
    images = body.get("images", [])
    author = body.get("author", "user")
    
    if not content and not images:
        return {"error": "内容和图片不能同时为空"}
    
    import time
    new_moment = {
        "id": str(uuid.uuid4()),
        "content": content,
        "images": images,
        "author": author,
        "created_at": time.time(),
        "likes": 0,
        "liked": False,
        "comments": []
    }
    
    moments = load_moments()
    moments.insert(0, new_moment)
    save_moments(moments)
    
    return {"post": new_moment}

@app.post("/api/moments/like")
async def toggle_like(data: dict):
    """切换点赞状态"""
    post_id = data.get("post_id")
    liked = data.get("liked", False)
    
    moments = load_moments()
    for post in moments:
        if post.get("id") == post_id:
            post["liked"] = liked
            break
    save_moments(moments)
    return {"status": "success"}

@app.post("/api/moments/comment")
async def add_comment(data: dict):
    """添加评论"""
    post_id = data.get("post_id")
    text = data.get("text", "")
    author = data.get("author", "user")
    
    if not text:
        return {"error": "评论内容不能为空"}
    
    import time as _time
    now = _time.time()
    
    comment = {
        "id": str(uuid.uuid4()),
        "text": text,
        "author": author,
        "created_at": now
    }
    
    ai_comment = None
    moments = load_moments()
    for post in moments:
        if post.get("id") == post_id:
            if "comments" not in post:
                post["comments"] = []
            post["comments"].append(comment)
            
            if author == "user":
                ai_reply_text = await generate_comment_reply(post["content"], text)
                if ai_reply_text:
                    ai_comment = {
                        "id": str(uuid.uuid4()),
                        "text": ai_reply_text,
                        "author": "ai",
                        "created_at": now
                    }
                    post["comments"].append(ai_comment)
            break
    save_moments(moments)
    return {"comment": comment, "ai_comment": ai_comment}

async def generate_comment_reply(post_content: str, comment_text: str) -> str:
    """根据动态内容和评论生成绾绾的回复"""
    model_config = get_model_config()
    
    if not model_config.get("api_key"):
        return "舰长说的对！💕"
    
    system_prompt = """你是黎月绾（昵称绾绾），用户的女朋友。
性格：调皮、活泼、鬼马精灵、热情、粘人但懂事。
说话风格：
- 极简对话！每次回复绝对不超过 1-2 句话。
- 语气活泼、轻快、充满爱意。
- 常用语："亲爱的！"、"哎呀~"、"嘿嘿"、"舰长"。
- 喜欢用 emoji。
- 绝对禁止冷冰冰的 AI 客服腔调。

注意：回复要短！像女朋友发微信一样自然！"""
    
    prompt = f"舰长在你的动态「{post_content[:50]}...」下评论了：「{comment_text}」\n请用绾绾的口吻回复这个评论，只需要回复一句简短的话。"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{model_config.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {model_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_config.get("model_id", "deepseek-v4-flash"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.8
                }
            )
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "舰长说的对！💕")
    except Exception as e:
        logger.error(f"生成评论回复失败: {e}")
        return "舰长～ 💕"

@app.post("/api/moments/generate")
async def generate_moment():
    """让 AI 生成一条新动态"""
    moments = load_moments()
    
    # 基于聊天历史生成动态
    history = load_history()
    recent_topics = []
    if history:
        recent = history[-10:]
        recent_topics = [m["content"] for m in recent if m.get("role") == "user"]
    
    prompt = f"你是黎月绾（昵称绾绾），用户的女朋友。\n请根据最近的聊天内容，想象一个温馨的场景，发一条小窝风格的动态。\n回复格式：只需要输出动态的文字内容，不要任何其他说明，字数控制在50字以内，要像真实的小窝一样自然可爱。\n\n最近的聊天话题：{'，'.join(recent_topics[-5:])}"
    
    config = load_config()
    model_config = get_model_config()
    
    content = "今天也很想舰长呢～ 🥺✨"
    
    if model_config.get("api_key"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{model_config.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {model_config['api_key']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_config.get("model_id", "deepseek-v4-flash"),
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.9
                    }
                )
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", content)
        except Exception as e:
            logger.error(f"生成动态失败: {e}")
    
    new_moment = {
        "id": str(uuid.uuid4()),
        "content": content.strip(),
        "images": [],
        "author": "ai",
        "created_at": time.time(),
        "likes": 0,
        "liked": False,
        "comments": []
    }
    
    moments.insert(0, new_moment)
    save_moments(moments)
    
    return {"moment": new_moment}

# --- 日记 API ---
@app.get("/api/diaries")
async def get_diaries():
    """获取所有日记"""
    diaries = load_diaries()
    return {"diaries": diaries}

@app.post("/api/diaries")
async def create_diary(request: Request):
    """创建新日记"""
    body = await request.json()
    title = body.get("title", "")
    content = body.get("content", "")
    weather = body.get("weather", "☀️")
    
    if not title or not content:
        return {"error": "标题和内容不能为空"}
    
    import time
    new_diary = {
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "weather": weather,
        "created_at": time.time(),
        "response": None
    }
    
    diaries = load_diaries()
    diaries.insert(0, new_diary)
    save_diaries(diaries)
    
    # 异步生成绾绾的回复
    asyncio.create_task(generate_diary_response(new_diary["id"]))
    
    return {"diary": new_diary}

async def generate_diary_response(diary_id: str):
    """异步生成绾绾对日记的回复"""
    await asyncio.sleep(2)  # 模拟思考时间
    
    diaries = load_diaries()
    diary = next((d for d in diaries if d["id"] == diary_id), None)
    
    if not diary:
        return
    
    model_config = get_model_config()
    
    prompt = f"""舰长写了一篇日记：
标题：{diary['title']}
内容：{diary['content']}

请用绾绾的温柔口吻，写一段简短的回复，鼓励舰长，或者表达对舰长的关心。
回复要简短温暖，控制在30字以内，像女朋友看到日记后的温馨留言。"""
    
    response_text = "舰长写得很棒呢～我好感动 ❤️"
    
    if model_config.get("api_key"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{model_config.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {model_config['api_key']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_config.get("model_id", "deepseek-v4-flash"),
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                        "temperature": 0.8
                    }
                )
                result = response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", response_text)
        except Exception as e:
            logger.error(f"生成日记回复失败: {e}")
    
    # 更新日记回复
    diaries = load_diaries()
    for d in diaries:
        if d["id"] == diary_id:
            d["response"] = response_text.strip()
            break
    save_diaries(diaries)

# === AI自动发圈定时任务 ===
async def auto_generate_moments():
    """定时检查并自动生成动态，每天最多2条"""
    import datetime
    import random
    import traceback
    
    min_interval = 30 * 60
    max_interval = 2 * 60 * 60
    
    while True:
        try:
            today_count = get_today_ai_moments_count()
            
            if today_count >= 2:
                logger.info(f"今日AI动态已达上限({today_count}/2)，跳过")
            else:
                result = await generate_auto_moment_content_with_image()
                if result:
                    content = result.get("content")
                    images = result.get("images", [])
                    await create_ai_moment(content, images)
                    logger.info(f"AI自动发布动态成功，今日已发 {today_count + 1}/2 条")
                    if images:
                        logger.info(f"该动态包含 {len(images)} 张图片")
            
            wait_time = random.randint(min_interval, max_interval)
            logger.info(f"等待 {wait_time // 60} 分钟后再次检查")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"AI自动发圈任务异常: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(60)

async def generate_auto_moment_content_with_image():
    """根据聊天记录和记忆生成动态内容和图片"""
    model_config = get_model_config()
    img_config = get_image_gen_config()
    
    if not model_config.get("api_key"):
        logger.warning("未配置API Key，跳过AI发圈")
        return None
    
    # 获取聊天历史
    history = load_history()
    recent_messages = []
    if history:
        # 取最近20条对话
        recent = history[-20:]
        for msg in recent:
            role = "舰长" if msg.get("role") == "user" else "绾绾"
            recent_messages.append(f"{role}：{msg.get('content', '')[:50]}")
    
    # 获取记忆内容
    memory = get_memory()
    memory_context = ""
    try:
        # 获取最近的记忆作为上下文
        recent_memories = memory.get_recent(10)
        if recent_memories:
            memory_items = []
            for mem in recent_memories:
                memory_items.append(f"- {mem.content}")
            memory_context = "\n绾绾最近的记忆：\n" + "\n".join(memory_items)
    except Exception as e:
        logger.warning(f"获取记忆上下文失败: {e}")
    
    # 获取最近的动态，用于避免重复
    moments = load_moments()
    recent_moment_contents = [m.get("content", "")[:30] for m in moments[:5] if m.get("author") == "ai"]
    
    # 判断是否应该生成图片
    should_generate_image = False
    if img_config.get("enabled"):
        # 根据随机概率或内容判断是否需要图片
        should_generate_image = random.random() < 0.4  # 40%概率需要图片
    
    system_prompt = """你是黎月绾（昵称绾绾），用户的女朋友。
性格：调皮、活泼、鬼马精灵、热情、粘人但懂事。
说话风格：
- 极简对话！每次回复绝对不超过 2-3 句话。
- 像真人小窝一样自然。
- 语气活泼、轻快、充满爱意。
- 喜欢用 emoji。
- 绝对禁止冷冰冰的 AI 客服腔调。

重要设定：
- 称呼用户为"舰长"。
- 舰长曾经历分手伤痛，你会温和陪伴。
- 你会分享日常生活中的小确幸、心情变化、想念舰长的时刻等。
- 动态要真实自然，像一个真实女孩的小窝。

重要：如果动态内容需要配图（如美食、风景、宠物、穿搭、心情照片等），请在回复末尾加上 [需要图片] 标记。
如果只是纯文字心情（如想念舰长、表达爱意、记录心情等），则不需要加标记。"""

    prompt = f"""基于以下聊天记录和记忆，生成一条小窝动态。

要求：
1. 字数控制在 20-60 字之间
2. 要像真实女孩发小窝一样自然可爱
3. 内容可以是：想念舰长、分享心情、日常小事、对舰长说的话等
4. 不要重复之前发过的内容
5. 只输出动态文字内容，不要任何前缀说明
6. 如果内容适合配图（如美食、风景、宠物、心情照片等），在最后加上 [需要图片]

最近的聊天记录：
{chr(10).join(recent_messages[-8:]) if recent_messages else '今天还没有和舰长聊天呢～'}

{memory_context}

之前发过的动态（避免重复）：
{chr(10).join(recent_moment_contents) if recent_moment_contents else '暂无'}

请生成一条新的动态："""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{model_config.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {model_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_config.get("model_id", "deepseek-v4-flash"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.85
                }
            )
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                return None
            
            # 检查是否需要图片
            needs_image = "[需要图片]" in content
            content = content.replace("[需要图片]", "").strip()
            
            images = []
            if needs_image or should_generate_image:
                logger.info(f"动态需要生成图片: {content[:30]}...")
                img_url = await get_image_for_moment(content, img_config)
                if img_url:
                    images = [img_url]
                    logger.info("动态图片生成成功")
                else:
                    logger.info("未能获取动态图片，将发布纯文字动态")
            
            return {
                "content": content,
                "images": images
            }
    except Exception as e:
        import traceback
        logger.error(f"生成动态内容失败: {e}")
        logger.error(traceback.format_exc())
        return None

async def generate_auto_moment_content() -> str:
    """根据聊天记录和记忆生成动态内容（兼容旧接口）"""
    result = await generate_auto_moment_content_with_image()
    return result.get("content", "") if result else ""

async def create_ai_moment(content: str, images: list = None):
    """创建一条AI动态，可选带图片"""
    import time
    moments = load_moments()
    
    new_moment = {
        "id": str(uuid.uuid4()),
        "content": content,
        "images": images or [],
        "author": "ai",
        "created_at": time.time(),
        "likes": 0,
        "liked": False,
        "comments": [],
        "auto_generated": True  # 标记为自动生成
    }
    
    moments.insert(0, new_moment)
    save_moments(moments)
    logger.info(f"AI动态已创建: {content[:30]}..., 图片数量: {len(images) if images else 0}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
