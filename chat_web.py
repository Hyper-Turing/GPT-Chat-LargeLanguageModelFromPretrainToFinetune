"""
chat_web.py — SFT 模型 Web 聊天服务器

基于 NanoChat 的 chat_web.py 风格，为自定义 SFT 模型提供 Web 可视化界面。
使用 FastAPI + SSE 流式输出，单文件同时提供 UI 和 API。

启动方式:
    python chat_web.py
    python chat_web.py --port 8080
    python chat_web.py --no-lora               # 不加载 LoRA
    python chat_web.py --lora-path out/sft/lora_best.pt

端点:
    GET  /                  - 聊天 UI
    POST /chat/completions  - 聊天 API（流式）
    GET  /health            - 健康检查
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description="SFT Model Web Chat Server")
parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or path")
parser.add_argument("--lora-path", type=str, default="out/sft/lora_best.pt", help="Path to LoRA weights")
parser.add_argument("--no-lora", action="store_true", help="Disable LoRA loading")
parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("-t", "--temperature", type=float, default=0.3, help="Default temperature")
parser.add_argument("-k", "--top-k", type=int, default=40, help="Default top-k")
parser.add_argument("--top-p", type=float, default=0.9, help="Default top-p")
parser.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max new tokens")
parser.add_argument("-p", "--port", type=int, default=8000, help="Server port")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
args = parser.parse_args()

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 请求限制
# ============================================================
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

# ============================================================
# 设备与精度
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DTYPE = DTYPE_MAP[args.dtype]

LORA_TARGETS = {"c_q", "c_v", "c_k", "c_proj", "lm_head"}


# ============================================================
# 模型加载
# ============================================================
def load_model():
    from transformers import AutoTokenizer
    from gpt_old import GPT
    from lora import apply_lora, load_lora_weights

    print(f"加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"加载 base 模型: {args.base_model}")
    model = GPT.from_pretrained(args.base_model)

    if not args.no_lora:
        print(f"  注入 LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, targets=LORA_TARGETS)
        print(f"  加载 LoRA 权重: {args.lora_path}")
        load_lora_weights(model, args.lora_path)
    else:
        print("  跳过 LoRA（--no-lora）")

    model = model.to(DTYPE).to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数: {total:.1f}M, 设备: {DEVICE}, 精度: {args.dtype}")
    return model, tokenizer


# ============================================================
# 生成锁（单模型串行推理）
# ============================================================
generate_lock = asyncio.Lock()


# ============================================================
# 流式生成
# ============================================================
def generate_tokens_sync(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, top_p):
    """
    同步生成 token，逐 token yield。
    因为 model.generate 可能不支持逐 token 回调，
    这里先尝试逐 token 生成，如果不行则一次性生成后模拟流式。
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    input_len = input_ids.shape[1]

    # 尝试一次性生成，然后逐 token 返回（模拟流式）
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            do_sample=True if temperature > 0 else False,
            eos_token_id=[im_end_id],
            pad_token_id=tokenizer.eos_token_id,
            use_kv_cache=True,
        )

    new_ids = output[0][input_len:]

    # 截断到第一个 <|im_end|>
    stop_position = len(new_ids)
    matches = (new_ids == im_end_id).nonzero(as_tuple=True)[0]
    if len(matches) > 0:
        stop_position = matches[0].item()
    new_ids = new_ids[:stop_position]

    return new_ids.tolist()


async def generate_stream(
    model, tokenizer, messages, temperature, max_new_tokens, top_k, top_p
) -> AsyncGenerator[str, None]:
    """异步流式生成（在线程池中运行同步推理）。"""

    # 构建 prompt
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)],
        dtype=torch.long,
        device=DEVICE,
    )

    logger.info(f"Prompt tokens: {input_ids.shape[1]}")

    # 在线程池中运行推理
    loop = asyncio.get_event_loop()
    token_ids = await loop.run_in_executor(
        None,
        generate_tokens_sync,
        model, tokenizer, input_ids,
        max_new_tokens, temperature, top_k, top_p,
    )

    # 逐 token 流式返回（模拟逐 token 效果）
    accumulated = []
    last_text = ""
    for tid in token_ids:
        accumulated.append(tid)
        current_text = tokenizer.decode(accumulated, skip_special_tokens=True)
        if not current_text.endswith("�"):
            new_text = current_text[len(last_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                last_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


# ============================================================
# 数据模型
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="至少需要一条消息")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"消息过多，最多 {MAX_MESSAGES_PER_REQUEST} 条")

    total_length = 0
    for i, msg in enumerate(request.messages):
        if not msg.content:
            raise HTTPException(status_code=400, detail=f"消息 {i} 内容为空")
        if len(msg.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"消息 {i} 过长（最多 {MAX_MESSAGE_LENGTH} 字符）")
        total_length += len(msg.content)

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"对话总长度过长（最多 {MAX_TOTAL_CONVERSATION_LENGTH} 字符）")

    for i, msg in enumerate(request.messages):
        if msg.role not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail=f"消息 {i} 的 role 无效")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"temperature 需在 {MIN_TEMPERATURE}~{MAX_TEMPERATURE} 之间")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k 需在 {MIN_TOP_K}~{MAX_TOP_K} 之间")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens 需在 {MIN_MAX_TOKENS}~{MAX_MAX_TOKENS} 之间")


# ============================================================
# FastAPI App
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  SFT Chat Server 启动中...")
    print("=" * 50)
    model, tokenizer = load_model()
    app.state.model = model
    app.state.tokenizer = tokenizer
    print(f"\n  服务已就绪: http://localhost:{args.port}")
    print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 50)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UI 页面（内联 HTML）
# ============================================================
@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


# ============================================================
# Chat API
# ============================================================
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)

    logger.info("=" * 30)
    for msg in request.messages:
        logger.info(f"[{msg.role.upper()}]: {msg.content}")
    logger.info("-" * 30)

    model = app.state.model
    tokenizer = app.state.tokenizer

    temperature = request.temperature if request.temperature is not None else args.temperature
    max_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens
    top_k = request.top_k if request.top_k is not None else args.top_k
    top_p = request.top_p if request.top_p is not None else args.top_p

    # 构建 messages（加入 system prompt）
    api_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in request.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    response_tokens = []

    async def stream_response():
        async with generate_lock:
            try:
                async for chunk in generate_stream(
                    model, tokenizer, api_messages,
                    temperature, max_tokens, top_k, top_p
                ):
                    # 记录 response
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if "token" in data:
                            response_tokens.append(data["token"])
                    except Exception:
                        pass
                    yield chunk
            finally:
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT]: {full_response}")
                logger.info("=" * 30)

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/health")
async def health():
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {
        "status": "ok",
        "ready": model_loaded,
        "device": DEVICE,
        "dtype": args.dtype,
        "base_model": args.base_model,
        "lora": not args.no_lora,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
