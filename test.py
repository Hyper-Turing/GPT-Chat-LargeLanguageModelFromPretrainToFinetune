import torch
from transformers import AutoTokenizer
from gpt import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-1.5B"   # 用 0.5B 更快验证

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Loading YOUR GPT (from_pretrained)...")
model = GPT.from_pretrained(model_name)
model = model.to(device)
model.eval()

# ===== 测试生成 =====
prompt = "你好，请介绍一下人工智能。"

input_ids = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids["input_ids"],
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n===== Your GPT Output =====\n")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


# ===== 对比 HF 原生 =====
from transformers import AutoModelForCausalLM

print("\nLoading HF original model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to(device)
hf_model.eval()

with torch.no_grad():
    hf_out = hf_model.generate(
        **input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

print("\n===== HF Model Output =====\n")
print(tokenizer.decode(hf_out[0], skip_special_tokens=True))
