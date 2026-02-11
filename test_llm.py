"""
自定义 GPT 模型推理脚本（适配 gpt.py 中的 GPT 类）

加载 out/sft/best_model 中的自定义 GPT checkpoint 进行交互式对话
"""

import torch
from transformers import AutoTokenizer
from gpt import GPT, GPTConfig
from lora import apply_lora, load_lora_weights

# ============================================================
#  配置
# ============================================================
# 基座模型（用于加载 tokenizer + 预训练权重）
BASE_MODEL = "Qwen/Qwen2.5-1.5B"

# LoRA 模式: 加载 HF base + lora_best.pt（当前训练方式）
# 非 LoRA 模式: 设 USE_LORA=False, 设 CKPT_PATH 为 best_model.pt / final_model.pt
USE_LORA = True
LORA_PATH = "out/sft/lora_best.pt"  # 或 lora_final.pt

# 非 LoRA 时的 checkpoint 路径（USE_LORA=True 时此项无效）
CKPT_PATH = None  # 例如 "out/sft/best_model.pt"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = {"c_q", "c_v"}

# 生成参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16


# ============================================================
#  模型加载
# ============================================================
def load_model():
    """
    LoRA 模式:
      1. 加载 base model
      2. 注入 LoRA 结构
      3. 加载 LoRA 权重
    """
    print(f"加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_LORA:
        # LoRA 模式: 加载 HF base → 注入 LoRA → 加载 LoRA 权重
        print(f"从 HF 加载 base 模型: {BASE_MODEL}")
        model = GPT.from_pretrained(BASE_MODEL)
        print(f"  注入 LoRA: r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
        apply_lora(model, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)
        print(f"  加载 LoRA 权重: {LORA_PATH}")
        load_lora_weights(model, LORA_PATH)
    elif CKPT_PATH is not None:
        # 非 LoRA: 加载完整 checkpoint {"config": ..., "state_dict": ...}
        print(f"加载自定义 GPT checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        print(f"  Config: {config}")
        model = GPT(config)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  state_dict loaded")
    else:
        # 测试 pipeline: 直接加载 HF 原始权重
        print(f"从 HF 加载预训练权重: {BASE_MODEL}")
        model = GPT.from_pretrained(BASE_MODEL)

    model = model.to(DTYPE).to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  总参数: {total:.1f}M, 可训练: {trainable:.1f}M")

    return model, tokenizer


# ============================================================
#  对话接口
# ============================================================
def chat_once(prompt, model, tokenizer, history=None):
    """单次对话生成"""
    messages = []
    if history:
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    # apply_chat_template 自动处理 <|im_start|> 等标记
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=DEVICE)
    input_len = input_ids.shape[1]

    # print(f"  [debug] 输入长度: {input_len} tokens")
    stop_ids = [tokenizer.eos_token_id]
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    # 用 GPT 自带的 generate 方法
    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=True,  # 启用采样
        eos_token_id=stop_ids,  # 传入 list
        pad_token_id=tokenizer.pad_token_id,
    )

    # 只解码新生成的部分
    new_ids = output[0][input_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)

    # 在 token id 层面截断：遇到 stop token 就截断
    for sid in stop_ids:
        if sid in new_ids:
            new_ids = new_ids[:new_ids.index(sid)]

    response = tokenizer.decode(new_ids, skip_special_tokens=True)

    # 二次兜底：截断残留的特殊标记文本
    for stop_tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        if stop_tag in response:
            response = response[:response.index(stop_tag)]

    return response.strip()


# ============================================================
#  测试
# ============================================================
def test_fixed_prompts(model, tokenizer):
    """固定 Prompt 测试"""
    print("\n" + "=" * 60)
    print("  固定 Prompt 测试")
    print("=" * 60)

    prompts = [
        "你好",
        "请用一句话介绍机器学习",
        "Hello, how are you?",
        "什么是Transformer？请简要说明。",
        "1+1等于几？",
        "写一个Python函数计算斐波那契数列",
    ]

    for prompt in prompts:
        response = chat_once(prompt, model, tokenizer)
        print(f"\n  User: {prompt}")
        print(f"  Assistant: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"  {'=' * 50}")


def test_multi_turn(model, tokenizer):
    """多轮对话测试"""
    print("\n" + "=" * 60)
    print("  多轮对话测试")
    print("=" * 60)

    history = []
    turns = [
        "你好，你叫什么名字？",
        "你能做什么？",
        "帮我写一个冒泡排序",
    ]
    for prompt in turns:
        response = chat_once(prompt, model, tokenizer, history)
        print(f"\n  User: {prompt}")
        print(f"  Assistant: {response[:300]}{'...' if len(response) > 300 else ''}")
        history.append((prompt, response))


def interactive_chat(model, tokenizer):
    """交互式对话"""
    print("\n" + "=" * 60)
    print("  交互式对话 (自定义 GPT)")
    print("  输入 quit 退出, clear 清空历史, test 跑固定测试")
    print("=" * 60)

    history = []
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
        if user_input.lower() == "test":
            test_fixed_prompts(model, tokenizer)
            continue

        response = chat_once(user_input, model, tokenizer, history)
        print(f"\nAssistant: {response}")

        history.append((user_input, response))
        if len(history) > 5:
            history = history[-5:]


def main():
    model, tokenizer = load_model()
    # 在 load_model() 之后
    text = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        # 注意：如果你的模型转成了 bf16，确保 idx 在同一设备
        logits, _ = model(idx)
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)  # 转 float 避免精度问题
        topk_vals, topk_ids = probs.topk(15)
        
        print("Top-15 下一个 token 预测:")
        for v, i in zip(topk_vals, topk_ids):
            t = tokenizer.decode([i.item()])
            print(f"  ID={i.item():6d}  prob={v.item():.4f}  text={repr(t)}")

    # 先跑固定测试
    # test_fixed_prompts(model, tokenizer)

    # 可选：多轮对话测试
    # test_multi_turn(model, tokenizer)

    # 进入交互模式
    interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()