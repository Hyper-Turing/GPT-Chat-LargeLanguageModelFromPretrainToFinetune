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
#  推理
# ============================================================
def apply_repetition_penalty(logits, generated_ids, penalty):
    """对已生成过的 token 施加重复惩罚, logits shape: [vocab_size]"""
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits
    unique_ids = list(set(generated_ids))
    score = logits[unique_ids]
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[unique_ids] = score
    return logits


def detect_repetition(ids, min_pattern=8, max_check=100):
    """检测最近生成的 token 是否陷入循环"""
    if len(ids) < min_pattern * 2:
        return False
    recent = ids[-max_check:]
    for plen in range(min_pattern, len(recent) // 2 + 1):
        pattern = recent[-plen:]
        prev = recent[-2 * plen:-plen]
        if pattern == prev:
            return True
    return False


def trim_verbose(text):
    """截断重复/废话尾巴：在最后一个完整句结束处截断"""
    if len(text) < 80:
        return text

    # 找重复段落
    for sep in ['。', '！', '？', '\n']:
        text = text.replace(sep, sep + '\x00')
    parts = [s.strip() for s in text.split('\x00') if s.strip()]

    seen = set()
    result = []
    for part in parts:
        key = part.replace('，', '').replace('。', '').replace('！', '').replace('？', '').strip()
        if len(key) > 10 and key in seen:
            break
        if len(key) > 10:
            seen.add(key)
        result.append(part)

    trimmed = ''.join(result)

    cut_patterns = [
        "如果您还有", "如果你还有", "如果有任何", "希望我能",
        "请随时告诉", "感谢您的", "祝您好运", "期待下次",
        "如果您想", "如果你想讨论", "欢迎继续", "请继续提问",
    ]
    for pat in cut_patterns:
        idx = trimmed.find(pat)
        if idx > 20:
            trimmed = trimmed[:idx].rstrip('，,、 ')
            break

    return trimmed.strip()


def sample_token(logits_2d):
    """从 [1, vocab] 的 logits 中采样一个 token，返回 [1, 1] tensor"""
    if TEMPERATURE > 0:
        logits_2d = logits_2d / TEMPERATURE

    if TOP_K > 0:
        v, _ = torch.topk(logits_2d, min(TOP_K, logits_2d.size(-1)))
        logits_2d[logits_2d < v[:, [-1]]] = -float("Inf")

    if TOP_P < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits_2d, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > TOP_P
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        indices_to_remove = remove.scatter(1, sorted_idx, remove)
        logits_2d[indices_to_remove] = -float("Inf")

    probs = torch.softmax(logits_2d, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # [1, 1]


def generate_reply(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=DEVICE)
    inp_len = ids.shape[1]

    stop_ids = [tokenizer.eos_token_id]
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    generated = []
    with lock:
        if USE_KV_CACHE:
            # ---- KV Cache 模式 ----
            total_len = inp_len + MAX_NEW_TOKENS
            kv_cache = model._create_kv_cache(1, total_len, DEVICE, DTYPE)

            # Prefill: 一次性处理所有输入 token
            logits, _ = model(ids, kv_cache=kv_cache)

            for _ in range(MAX_NEW_TOKENS):
                next_logits = logits[:, -1, :]  # [1, vocab]

                # 重复惩罚
                next_logits[0] = apply_repetition_penalty(next_logits[0], generated, REPETITION_PENALTY)

                # 采样
                next_id = sample_token(next_logits)  # [1, 1]
                token_id = next_id.item()
                generated.append(token_id)

                # EOS 检查
                if token_id in stop_ids:
                    break

                # 重复循环检测
                if len(generated) > 30 and detect_repetition(generated):
                    break

                # Decode step: 只送入新 token，KV Cache 自动拼接历史
                logits, _ = model(next_id, kv_cache=kv_cache)

        else:
            # ---- 无 Cache 模式（原始实现）----
            for _ in range(MAX_NEW_TOKENS):
                idx_cond = ids if ids.size(1) <= model.config.max_seq_len else ids[:, -model.config.max_seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]  # [1, vocab]

                # 重复惩罚
                logits[0] = apply_repetition_penalty(logits[0], generated, REPETITION_PENALTY)

                # 采样
                next_id = sample_token(logits)  # [1, 1]
                ids = torch.cat([ids, next_id], dim=1)

                token_id = next_id.item()
                generated.append(token_id)

                if token_id in stop_ids:
                    break

                if len(generated) > 30 and detect_repetition(generated):
                    break

    # 截断 stop token
    for sid in stop_ids:
        if sid in generated:
            generated = generated[:generated.index(sid)]

    resp = tokenizer.decode(generated, skip_special_tokens=True)
    for tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        if tag in resp:
            resp = resp[:resp.index(tag)]
    resp = trim_verbose(resp)
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
  body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: #ffffff; 
    color: #374151; 
    height: 100vh; 
    display: flex; 
    flex-direction: column; 
  }
  
  .header { 
    padding: 16px 20px; 
    background: #ffffff; 
    border-bottom: 1px solid #e5e7eb; 
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  
  .header h1 { 
    font-size: 20px; 
    color: #374151;
    font-weight: 600;
  }
  
  .header p { 
    font-size: 13px; 
    color: #6b7280; 
    margin-top: 2px; 
  }
  
  #chat { 
    flex: 1; 
    overflow-y: auto; 
    padding: 20px; 
    background: #f9fafb;
    display: flex; 
    flex-direction: column; 
    gap: 8px;  /* 减小间距 */
  }
  
  .msg { 
    max-width: 85%; 
    padding: 12px 16px; 
    border-radius: 12px; 
    line-height: 1.5; 
    font-size: 15px; 
    white-space: pre-wrap; 
    word-break: break-word; 
    animation: fadeIn .3s ease;
  }
  
  .user { 
    align-self: flex-end; 
    background: #3b82f6; 
    color: white; 
    margin-left: auto;
    border-radius: 12px 12px 4px 12px;
  }
  
  .bot { 
    align-self: flex-start; 
    background: white; 
    color: #374151; 
    border: 1px solid #e5e7eb;
    border-radius: 12px 12px 12px 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  
  .bot pre { 
    background: #f3f4f6; 
    padding: 10px 12px; 
    border-radius: 6px; 
    overflow-x: auto; 
    margin: 8px 0; 
    font-size: 14px; 
    border: 1px solid #e5e7eb;
  }
  
  .bot code { 
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace; 
    font-size: 13.5px; 
    color: #dc2626;
    background: #fef2f2;
    padding: 2px 4px;
    border-radius: 3px;
  }
  
  .typing { 
    align-self: flex-start; 
    padding: 12px 16px; 
    background: white; 
    border: 1px solid #e5e7eb; 
    border-radius: 12px 12px 12px 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .typing-dots {
    display: flex;
    gap: 4px;
  }
  
  .typing span { 
    display: inline-block; 
    width: 6px; 
    height: 6px; 
    background: #9ca3af; 
    border-radius: 50%; 
    animation: pulse 1.4s infinite ease-in-out;
  }
  
  .typing span:nth-child(1) { animation-delay: -0.32s; }
  .typing span:nth-child(2) { animation-delay: -0.16s; }
  
  @keyframes pulse {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
  }
  
  @keyframes fadeIn { 
    from { opacity: 0; transform: translateY(4px); } 
    to { opacity: 1; transform: translateY(0); } 
  }
  
  .input-area { 
    padding: 16px 20px; 
    background: white; 
    border-top: 1px solid #e5e7eb; 
    display: flex; 
    gap: 10px; 
    align-items: flex-end;
  }
  
  #input { 
    flex: 1; 
    padding: 12px 16px; 
    border-radius: 12px; 
    border: 1px solid #d1d5db; 
    background: white; 
    color: #374151; 
    font-size: 15px; 
    outline: none; 
    resize: none; 
    max-height: 120px; 
    font-family: inherit;
    transition: border-color 0.2s;
  }
  
  #input:focus { 
    border-color: #3b82f6; 
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
  
  #input::placeholder {
    color: #9ca3af;
  }
  
  #send { 
    padding: 0 20px; 
    height: 42px;
    border-radius: 12px; 
    border: none; 
    background: #3b82f6; 
    color: white; 
    font-size: 14px; 
    font-weight: 500;
    cursor: pointer; 
    transition: background-color 0.2s;
  }
  
  #send:hover { 
    background: #2563eb; 
  }
  
  #send:disabled { 
    background: #9ca3af; 
    cursor: not-allowed; 
  }
  
  #clear { 
    padding: 0 16px; 
    height: 42px;
    border-radius: 12px; 
    border: 1px solid #d1d5db; 
    background: white; 
    color: #6b7280; 
    font-size: 14px; 
    cursor: pointer; 
    transition: all 0.2s;
  }
  
  #clear:hover { 
    border-color: #3b82f6; 
    color: #3b82f6; 
    background: #f8fafc;
  }
  
  /* 滚动条样式 */
  #chat::-webkit-scrollbar {
    width: 6px;
  }
  
  #chat::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }
  
  #chat::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  #chat::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
  
  /* 响应式调整 */
  @media (max-width: 768px) {
    .msg { max-width: 90%; }
    #input { font-size: 16px; } /* 移动端输入法优化 */
  }
</style>
</head>
<body>
<div class="header">
  <h1>🤖 GPT Chat</h1>
  <p>基于 Qwen2.5-1.5B + LoRA SFT 微调</p>
</div>
<div id="chat"></div>
<div class="input-area">
  <textarea id="input" rows="1" placeholder="输入消息..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button id="send" onclick="send()">发送</button>
  <button id="clear" onclick="clearChat()">清空</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let history = [];

input.addEventListener('input', function() {
  this.style.height = 'auto';
  const maxHeight = 160; // 稍微增加最大高度
  this.style.height = Math.min(this.scrollHeight, maxHeight) + 'px';
});

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  
  // 处理代码块
  let html = text
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
  
  d.innerHTML = html;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function showTyping() {
  const d = document.createElement('div');
  d.className = 'typing';
  d.id = 'typing';
  
  const dots = document.createElement('div');
  dots.className = 'typing-dots';
  dots.innerHTML = '<span></span><span></span><span></span>';
  
  d.appendChild(dots);
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;
  sendBtn.textContent = '发送中...';

  addMsg('user', text);
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: history })
    });
    
    const data = await res.json();
    hideTyping();

    if (data.error) {
      addMsg('bot', '⚠️ ' + data.error);
    } else {
      addMsg('bot', data.reply);
      history.push([text, data.reply]);
      // 限制历史记录长度
      if (history.length > 10) history = history.slice(-10);
    }
  } catch(e) {
    hideTyping();
    addMsg('bot', '⚠️ 请求失败: ' + e.message);
    console.error('请求错误:', e);
  }
  
  sendBtn.disabled = false;
  sendBtn.textContent = '发送';
  input.focus();
}

function clearChat() {
  if (chat.children.length === 0) return;
  
  if (confirm('确定要清空对话记录吗？')) {
    history = [];
    chat.innerHTML = '';
    input.focus();
  }
}

// 页面加载后自动聚焦输入框
window.addEventListener('load', () => {
  input.focus();
});

// 支持 Ctrl+Enter 发送
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    e.preventDefault();
    send();
  }
});
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