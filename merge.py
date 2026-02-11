"""
在自定义 GPT 结构上合并 LoRA 权重，再导出为 HF 格式
"""

import torch
import os
from transformers import AutoTokenizer
from gpt import GPT
from lora import apply_lora, load_lora_weights, LoRALinear

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_PATH = "out/sft/lora_best.pt"
MERGED_DIR = "out/sft/merged_model"
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = {"c_q", "c_v"}  # 和训练时一致！

DEVICE = "cpu"

# 1. 加载 base 模型（自定义 GPT 结构）
print(f"加载 base 模型: {BASE_MODEL}")
model = GPT.from_pretrained(BASE_MODEL)

# 2. 注入 LoRA（和训练时完全相同的参数）
print(f"注入 LoRA: r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
apply_lora(model, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)

# 3. 加载 LoRA 权重
print(f"加载 LoRA 权重: {LORA_PATH}")
load_lora_weights(model, LORA_PATH)

# 4. 合并: 将 LoRA 权重合并回主权重
merged_count = 0
for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
        with torch.no_grad():
            # W = W + scaling * B @ A
            module.weight.data += module.scaling * (module.lora_B @ module.lora_A)
        merged_count += 1
        print(f"  合并: {name}")

print(f"共合并 {merged_count} 个 LoRA 层")

# 5. 导出为 HF 可加载的格式
#    需要将自定义 GPT 的 state_dict 映射回 HF 的 key 名称
#    这里直接保存为自定义格式，由 GPT.from_pretrained 加载
os.makedirs(MERGED_DIR, exist_ok=True)

# 方案 A: 保存为自定义 GPT checkpoint（推荐，最简单）
save_path = os.path.join(MERGED_DIR, "best_model.pt")
# 清理 LoRA 相关的参数，只保留主权重
state_dict = {}
for k, v in model.state_dict().items():
    if "lora_A" not in k and "lora_B" not in k:
        state_dict[k] = v

torch.save({
    "config": model.config,
    "state_dict": state_dict,
}, save_path)

# 保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_DIR)

print(f"\n合并模型已保存到: {save_path}")
print(f"Tokenizer 已保存到: {MERGED_DIR}")