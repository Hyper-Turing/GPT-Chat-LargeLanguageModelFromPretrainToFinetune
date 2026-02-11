"""
单文件网页聊天: Flask 后端 + 内嵌前端
启动: python web_chat.py
访问: http://localhost:5000
"""

import torch, json, threading
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer
from gpt import GPT
from lora import apply_lora, load_lora_weights
import os

# ============================================================
#  配置
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
MERGED_DIR = "out/sft/merged_model"
USE_LORA = True
LORA_PATH = "out/sft/lora_best.pt"
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = {"c_q", "c_v"}

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.5
USE_KV_CACHE = True  # 启用 KV Cache 加速推理

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

lock = threading.Lock()

# ============================================================
#  模型加载
# ============================================================
def load_model():
    merged_ckpt = os.path.join(MERGED_DIR, "best_model.pt")

    if os.path.exists(merged_ckpt):
        print(f"加载合并模型: {merged_ckpt}")
        tok = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
        ckpt = torch.load(merged_ckpt, map_location="cpu", weights_only=False)
        mdl = GPT(ckpt["config"])
        mdl.load_state_dict(ckpt["state_dict"])
    elif USE_LORA:
        print(f"加载 base + LoRA")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        mdl = GPT.from_pretrained(BASE_MODEL)
        apply_lora(mdl, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)
        load_lora_weights(mdl, LORA_PATH)
    else:
        print(f"加载 HF 原始权重")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        mdl = GPT.from_pretrained(BASE_MODEL)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = mdl.to(DTYPE).to(DEVICE).eval()
    total = sum(p.numel() for p in mdl.parameters()) / 1e6
    print(f"模型已加载: {total:.1f}M 参数, 设备: {DEVICE}, KV Cache: {'启用' if USE_KV_CACHE else '禁用'}")
    return mdl, tok

model, tokenizer = load_model()

# ============================================================
#  推理 (简化版，直接使用 gpt.generate)
# ============================================================
def generate_reply(messages):
    # 构造对话文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=DEVICE)
    
    # 准备 stop token IDs
    stop_ids = [tokenizer.eos_token_id]
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    with lock:
        # 直接使用 gpt.generate 生成回复
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_kv_cache=USE_KV_CACHE,
        )

    # 解码新生成的部分
    input_len = input_ids.shape[1]
    new_ids = output[0][input_len:].tolist()
    
    # 截断 stop token
    for sid in stop_ids:
        if sid in new_ids:
            new_ids = new_ids[:new_ids.index(sid)]
            break
    
    # 解码并清理
    resp = tokenizer.decode(new_ids, skip_special_tokens=True)
    
    # 截断特殊标记
    for tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "_typeDefinition", "_类型定义", "_类型解释", "typeDefinition"]:
        if tag in resp:
            resp = resp[:resp.index(tag)]
    
    # 清理可能的尾部垃圾
    resp = resp.rstrip(" _~`\"'")
    
    return resp.strip()

# ============================================================
#  Flask
# ============================================================
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }
.header { padding: 16px; background: #fff; border-bottom: 1px solid #e5e7eb; text-align: center; }
#chat { flex: 1; overflow-y: auto; padding: 20px 10%; background: #f9fafb; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 15px; white-space: pre-wrap; word-break: break-word; }
.user { align-self: flex-end; background: #3b82f6; color: #fff; border-radius: 12px 12px 4px 12px; }
.bot { align-self: flex-start; background: #fff; color: #374151; border: 1px solid #e5e7eb; border-radius: 12px 12px 12px 4px; }
.bot pre { background: #f3f4f6; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; font-size: 14px; }
.bot code { font-family: monospace; font-size: 13px; color: #dc2626; background: #fef2f2; padding: 2px 4px; border-radius: 3px; }
.typing { align-self: flex-start; padding: 12px 16px; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; display: flex; gap: 4px; }
.typing span { width: 6px; height: 6px; background: #9ca3af; border-radius: 50%; animation: p 1.4s infinite; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
@keyframes p { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }
.input-area { padding: 16px 10%; background: #fff; border-top: 1px solid #e5e7eb; display: flex; gap: 10px; }
#input { flex: 1; padding: 10px 16px; border-radius: 10px; border: 1px solid #d1d5db; font-size: 15px; outline: none; resize: none; min-height: 40px; max-height: 120px; }
#input:focus { border-color: #3b82f6; }
#send, #clear { padding: 10px 20px; border-radius: 10px; border: none; font-size: 14px; cursor: pointer; }
#send { background: #3b82f6; color: #fff; }
#send:disabled { background: #9ca3af; }
#clear { background: #fff; color: #6b7280; border: 1px solid #d1d5db; }
@media (max-width: 768px) { #chat, .input-area { padding: 16px 5%; } .msg { max-width: 85%; } }
</style>
</head>
<body>
<div class="header"><h3>GPT Chat</h3></div>
<div id="chat"></div>
<div class="input-area">
<textarea id="input" rows="1" placeholder="输入消息..."></textarea>
<button id="send" onclick="send()">发送</button>
<button id="clear" onclick="clearChat()">清空</button>
</div>
<script>
const chat = document.getElementById('chat'), input = document.getElementById('input'), sendBtn = document.getElementById('send');
let history = [];

input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 120) + 'px'; });
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

function fmt(text) {
  return text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\n/g, '<br>');
}
function add(role, text) {
  const d = document.createElement('div'); d.className = 'msg ' + role; d.innerHTML = fmt(text); chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
function typing() { const d = document.createElement('div'); d.className = 'typing'; d.id = 't'; d.innerHTML = '<span></span><span></span><span></span>'; chat.appendChild(d); chat.scrollTop = chat.scrollHeight; }
function hideTyping() { const el = document.getElementById('t'); if (el) el.remove(); }

async function send() {
  const text = input.value.trim(); if (!text) return;
  input.value = ''; input.style.height = 'auto'; sendBtn.disabled = true;
  add('user', text); typing();
  try {
    const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text, history }) });
    const data = await res.json(); hideTyping();
    if (data.error) add('bot', '⚠️ ' + data.error);
    else { add('bot', data.reply); history.push([text, data.reply]); if (history.length > 10) history = history.slice(-10); }
  } catch(e) { hideTyping(); add('bot', '⚠️ 请求失败'); }
  sendBtn.disabled = false; input.focus();
}

function clearChat() { if (!chat.children.length) return; if (confirm('清空对话?')) { history = []; chat.innerHTML = ''; input.focus(); } }
window.addEventListener('load', () => input.focus());
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML, content_type="text/html")

@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        data = request.json
        msg = data.get("message", "").strip()
        hist = data.get("history", [])
        if not msg:
            return jsonify({"error": "消息不能为空"})

        messages = []
        messages.append({
        "role": "system", 
        "content": "你是一个极度简洁的AI助手。规则：1.只回答核心内容，禁止废话；2.禁止重复和过度解释；3.每个回答控制在30~字以内；4.只陈述事实，适当添加寒暄。"
    })
        
        for u, b in hist:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
        messages.append({"role": "user", "content": msg})

        reply = generate_reply(messages)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  网页聊天已启动: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)