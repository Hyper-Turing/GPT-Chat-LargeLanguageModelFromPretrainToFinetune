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

MAX_NEW_TOKENS = 256   # 缩短，避免废话过多
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.5  # 加大惩罚

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
    print(f"模型已加载: {total:.1f}M 参数, 设备: {DEVICE}")
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
    # 如果文本较短不处理
    if len(text) < 80:
        return text

    # 找重复段落：如果某个句子片段重复出现，在第二次出现前截断
    sentences = []
    for sep in ['。', '！', '？', '\n']:
        text = text.replace(sep, sep + '\x00')
    parts = [s.strip() for s in text.split('\x00') if s.strip()]

    seen = set()
    result = []
    for part in parts:
        # 取核心内容（去掉标点）做去重key
        key = part.replace('，', '').replace('。', '').replace('！', '').replace('？', '').strip()
        if len(key) > 10 and key in seen:
            break  # 遇到重复句子就停
        if len(key) > 10:
            seen.add(key)
        result.append(part)

    trimmed = ''.join(result)

    # 额外：砍掉常见废话结尾模式
    cut_patterns = [
        "如果您还有", "如果你还有", "如果有任何", "希望我能",
        "请随时告诉", "感谢您的", "祝您好运", "期待下次",
        "如果您想", "如果你想讨论", "欢迎继续", "请继续提问",
    ]
    for pat in cut_patterns:
        idx = trimmed.find(pat)
        if idx > 20:  # 确保不是整段都被砍
            trimmed = trimmed[:idx].rstrip('，,、 ')
            break

    return trimmed.strip()


def generate_reply(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=DEVICE)
    inp_len = ids.shape[1]

    stop_ids = [tokenizer.eos_token_id]
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    # 手动逐 token 生成，以便加 repetition penalty + 重复检测
    generated = []
    with lock:
        for _ in range(MAX_NEW_TOKENS):
            idx_cond = ids if ids.size(1) <= model.config.max_seq_len else ids[:, -model.config.max_seq_len:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # [1, vocab]

            # 重复惩罚 (在 [vocab] 维度上操作)
            logits_1d = logits[0]  # [vocab]
            logits_1d = apply_repetition_penalty(logits_1d, generated, REPETITION_PENALTY)
            logits = logits_1d.unsqueeze(0)  # [1, vocab]

            # temperature
            if TEMPERATURE > 0:
                logits = logits / TEMPERATURE

            # top-k
            if TOP_K > 0:
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # top-p
            if TOP_P < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > TOP_P
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            ids = torch.cat([ids, next_id], dim=1)

            token_id = next_id.item()
            generated.append(token_id)

            # EOS 检查
            if token_id in stop_ids:
                break

            # 重复循环检测: 提前终止
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
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #0f0f1a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 16px 24px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-bottom: 1px solid #2a2a4a; text-align: center; }
  .header h1 { font-size: 20px; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header p { font-size: 12px; color: #888; margin-top: 4px; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 16px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-break: break-word; animation: fadeIn .3s; }
  .user { align-self: flex-end; background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; border-bottom-right-radius: 4px; }
  .bot { align-self: flex-start; background: #1e1e36; border: 1px solid #2a2a4a; border-bottom-left-radius: 4px; }
  .bot pre { background: #12121f; padding: 10px; border-radius: 8px; overflow-x: auto; margin: 8px 0; font-size: 13px; }
  .bot code { font-family: "Fira Code", monospace; font-size: 13px; }
  .typing { align-self: flex-start; padding: 12px 20px; background: #1e1e36; border: 1px solid #2a2a4a; border-radius: 16px; }
  .typing span { display: inline-block; width: 8px; height: 8px; background: #667eea; border-radius: 50%; margin: 0 2px; animation: bounce .6s infinite alternate; }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { to { transform: translateY(-8px); opacity: .4; } }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .input-area { padding: 16px 20px; background: #1a1a2e; border-top: 1px solid #2a2a4a; display: flex; gap: 10px; }
  #input { flex: 1; padding: 12px 16px; border-radius: 24px; border: 1px solid #2a2a4a; background: #12121f; color: #e0e0e0; font-size: 14px; outline: none; resize: none; max-height: 120px; }
  #input:focus { border-color: #667eea; }
  #send { padding: 0 24px; border-radius: 24px; border: none; background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; font-size: 14px; cursor: pointer; transition: opacity .2s; }
  #send:hover { opacity: .85; }
  #send:disabled { opacity: .4; cursor: not-allowed; }
  #clear { padding: 0 16px; border-radius: 24px; border: 1px solid #2a2a4a; background: transparent; color: #888; font-size: 13px; cursor: pointer; }
  #clear:hover { border-color: #667eea; color: #e0e0e0; }
</style>
</head>
<body>
<div class="header">
  <h1>🤖 GPT Chat</h1>
  <p>权重来自Qwen2.5-1.5B + 自定义模型 + LoRA SFT</p>
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

// 自动调整输入框高度
input.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  // 简单的 code block 渲染
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  d.innerHTML = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function showTyping() {
  const d = document.createElement('div');
  d.className = 'typing';
  d.id = 'typing';
  d.innerHTML = '<span></span><span></span><span></span>';
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
      if (history.length > 5) history = history.slice(-5);
    }
  } catch(e) {
    hideTyping();
    addMsg('bot', '⚠️ 请求失败: ' + e.message);
  }
  sendBtn.disabled = false;
  input.focus();
}

function clearChat() {
  history = [];
  chat.innerHTML = '';
}
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