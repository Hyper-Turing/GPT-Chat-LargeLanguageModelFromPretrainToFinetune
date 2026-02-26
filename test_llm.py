import torch
from transformers import AutoTokenizer
from gpt import GPT
from lora import apply_lora, load_lora_weights

# ============================================================
# 配置
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B"

USE_LORA = True
LORA_PATH = "out/sft/lora_best.pt"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = {"c_q", "c_v", "c_k", "c_proj", "lm_head"}

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_K = 40
TOP_P = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


# ============================================================
# 模型加载
# ============================================================
def load_model():
    print(f"加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if USE_LORA:
        print(f"加载 base 模型: {BASE_MODEL}")
        model = GPT.from_pretrained(BASE_MODEL)
        print(f"  注入 LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
        apply_lora(model, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)
        print(f"  加载 LoRA 权重: {LORA_PATH}")
        load_lora_weights(model, LORA_PATH)
    else:
        model = GPT.from_pretrained(BASE_MODEL)

    model = model.to(DTYPE).to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数: {total:.1f}M")
    return model, tokenizer


# ============================================================
# 对话接口
# ============================================================
def chat_once(prompt, model, tokenizer, history=None, debug=False):
    messages = []
    if history:
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)],
        dtype=torch.long, device=DEVICE
    )
    input_len = input_ids.shape[1]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [im_end_id]

    if debug:
        print(f"\n{'='*60}")
        print(f"Prompt ({input_len} tokens):")
        print(text)
        print(f"Stop IDs: {stop_ids} (<|im_end|>={im_end_id})")
        print(f"{'='*60}")

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_ids = output[0][input_len:]

    if debug:
        print(f"\n生成的 Token IDs ({len(new_ids)} tokens):")
        print(f"  {new_ids.tolist()[:30]}{'...' if len(new_ids) > 30 else ''}")
        print(f"  解码: {repr(tokenizer.decode(new_ids, skip_special_tokens=False)[:200])}")

    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================
# 测试
# ============================================================
def chat(model, tokenizer):
    print("\n" + "=" * 60)
    print("  quit=退出, clear=清空历史, debug=切换调试, test=固定测试")
    print("=" * 60)

    history = []
    debug_mode = False

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print("  历史已清空")
            continue
        if user_input.lower() == "debug":
            debug_mode = not debug_mode
            print(f"  调试模式: {'开启' if debug_mode else '关闭'}")
            continue

        response = chat_once(user_input, model, tokenizer, history, debug=debug_mode)
        print(f"\nAssistant: {response}")

        history.append((user_input, response))
        if len(history) > 5:
            history = history[-5:]


def main():
    model, tokenizer = load_model()
    chat(model, tokenizer)


if __name__ == "__main__":
    main()