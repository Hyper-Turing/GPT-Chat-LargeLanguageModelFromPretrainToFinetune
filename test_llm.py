"""
test_llm.py — SFT 模型推理测试
"""

import torch
from transformers import AutoTokenizer
from gpt import GPT
from lora import apply_lora, load_lora_weights

# ============================================================
# 配置
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B" # 改用0.5B 加速训练

USE_LORA = True
LORA_PATH = "out/sft/lora_best.pt"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = {"c_q", "c_v", "c_k", "c_proj", "lm_head"} # 一定要加lm_head

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_K = 40
TOP_P = 0.9
USE_KV_CACHE = True

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
    """单次对话生成"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    if history:
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    # 使用 apply_chat_template — 与训练时完全一致
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)],
        dtype=torch.long, device=DEVICE
    )
    input_len = input_ids.shape[1]

    # ============================================================
    # 关键：只用 <|im_end|> 作为 stop token
    # 模型通过 SFT 学到的停止信号就是 <|im_end|> (151645)
    # 绝对不要把 <|endoftext|> (151643) 加进来
    # ============================================================
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
        use_kv_cache=USE_KV_CACHE,
    )

    new_ids = output[0][input_len:]

    if debug:
        print(f"\n生成的 Token IDs ({len(new_ids)} tokens):")
        print(f"  {new_ids.tolist()[:30]}{'...' if len(new_ids) > 30 else ''}")
        print(f"  解码: {repr(tokenizer.decode(new_ids, skip_special_tokens=False)[:200])}")

        # 检查是否生成了 <|im_end|>
        if im_end_id in new_ids.tolist():
            pos = new_ids.tolist().index(im_end_id)
            print(f"  \033[92m✓ 在位置 {pos} 生成了 <|im_end|>\033[0m")
        else:
            print(f"  \033[91m✗ 未生成 <|im_end|>，跑满了 {len(new_ids)} tokens\033[0m")

    # 截断到第一个 stop token
    stop_position = len(new_ids)
    for sid in stop_ids:
        matches = (new_ids == sid).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            first_match = matches[0].item()
            if first_match < stop_position:
                stop_position = first_match
    new_ids = new_ids[:stop_position]

    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================
# 测试
# ============================================================
def test_fixed_prompts(model, tokenizer):
    print("\n" + "=" * 60)
    print("  固定 Prompt 测试")
    print("=" * 60)

    prompts = [
        "你好",
        "请用一句话介绍机器学习",
        "Hello, how are you?",
        "1+1等于几？",
    ]

    for prompt in prompts:
        response = chat_once(prompt, model, tokenizer, debug=True)
        print(f"\n  User: {prompt}")
        print(f"  Assistant: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"  {'─' * 50}")


def interactive_chat(model, tokenizer):
    print("\n" + "=" * 60)
    print("  交互式对话")
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
        if user_input.lower() == "test":
            test_fixed_prompts(model, tokenizer)
            continue

        response = chat_once(user_input, model, tokenizer, history, debug=debug_mode)
        print(f"\nAssistant: {response}")

        history.append((user_input, response))
        if len(history) > 5:
            history = history[-5:]


def main():
    model, tokenizer = load_model()
    interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()