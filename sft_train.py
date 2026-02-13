# """
# sft_train.py — 使用自定义 GPT 模型做 SFT 微调

# 改动点 (相对 HuggingFace 版):
#   1. 用 GPT.from_pretrained() 加载模型
#   2. 前向调用: model(idx, targets) → (logits, loss)
#   3. LoRA targets 改为自定义层名: c_q, c_v (而非 q_proj, v_proj)
#   4. 保存/加载用 torch.save / torch.load
  
# 已弃用 已弃用 已弃用 已弃用  已弃用

# """

# import os
# import math
# import json
# import random
# from contextlib import nullcontext

# import torch
# from transformers import AutoTokenizer
# from gpt import GPT, GPTConfig
# from lora import apply_lora, mark_only_lora_as_trainable, lora_state_dict, load_lora_weights
# from torch.utils.data import Dataset

# # =============================================================================
# # 配置
# # =============================================================================

# # 模型（显存不够就换 0.5B）
# MODEL_NAME = "Qwen/Qwen2.5-1.5B"
# # MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# # 数据
# SFT_DATA_PATH = "data/sft/sft_data.jsonl"
# TRAIN_PATH = "data/sft/sft_train.jsonl"
# VAL_PATH = "data/sft/sft_val.jsonl"
# VAL_RATIO = 0.01

# # 输出
# OUT_DIR = "out/sft"

# # LoRA
# USE_LORA = True
# LORA_R = 16
# LORA_ALPHA = 32
# LORA_DROPOUT = 0.05
# LORA_TARGETS = {"c_q", "c_v"}  # 对应 HF 的 q_proj, v_proj

# # 训练超参
# MAX_SEQ_LEN = 512
# BATCH_SIZE = 2
# GRAD_ACCUM_STEPS = 4
# MAX_ITERS = 3000
# EVAL_INTERVAL = 200
# LOG_INTERVAL = 10
# LEARNING_RATE = 2e-4 if USE_LORA else 2e-5
# MIN_LR = 2e-6
# WARMUP_ITERS = 100
# WEIGHT_DECAY = 0.01
# GRAD_CLIP = 1.0
# PATIENCE = 8

# # 设备
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = "bfloat16" if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else "float16"

# # wandb
# WANDB_LOG = True
# WANDB_PROJECT = "qwen-sft"
# WANDB_RUN_NAME = f"sft-{'lora' if USE_LORA else 'full'}-{MODEL_NAME.split('/')[-1]}-custom"


# # =============================================================================
# # 数据集
# # =============================================================================

# class SFTDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_seq_len):
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
#         self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

#         with open(data_path, 'r', encoding='utf-8') as f:
#             self.data = [json.loads(line) for line in f if line.strip()]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         if "conversations" in item:
#             messages = item["conversations"]
#         elif "messages" in item:
#             messages = item["messages"]
#         else:
#             messages = [
#                 {"role": "user", "content": item.get("instruction", "") + item.get("input", "")},
#                 {"role": "assistant", "content": item.get("output", "")},
#             ]

#         input_ids, labels = self._encode_with_mask(messages)

#         if len(input_ids) > self.max_seq_len:
#             input_ids = input_ids[:self.max_seq_len]
#             labels = labels[:self.max_seq_len]

#         pad_len = self.max_seq_len - len(input_ids)
#         if pad_len > 0:
#             input_ids = input_ids + [self.pad_id] * pad_len
#             labels = labels + [-100] * pad_len

#         return (
#             torch.tensor(input_ids, dtype=torch.long),
#             torch.tensor(labels, dtype=torch.long),
#         )

#     def _encode_with_mask(self, messages):
#         input_ids = []
#         labels = []

#         for i, msg in enumerate(messages):
#             partial_msgs = messages[:i + 1]
#             is_last_user = (msg["role"] != "assistant")
#             result = self.tokenizer.apply_chat_template(
#                 partial_msgs, tokenize=True,
#                 add_generation_prompt=is_last_user,
#                 return_tensors=None,
#             )

#             if hasattr(result, 'input_ids'):
#                 raw_ids = result['input_ids']
#                 cur_ids = raw_ids.tolist() if hasattr(raw_ids, 'tolist') else list(raw_ids)
#             elif isinstance(result, list):
#                 cur_ids = result
#             else:
#                 cur_ids = list(result)

#             cur_ids = [int(x) for x in cur_ids]
#             new_ids = cur_ids[len(input_ids):]

#             if msg["role"] == "assistant":
#                 input_ids.extend(new_ids)
#                 labels.extend(new_ids)
#             else:
#                 input_ids.extend(new_ids)
#                 labels.extend([-100] * len(new_ids))

#         eos = self.tokenizer.eos_token_id
#         if eos is not None and (len(input_ids) == 0 or input_ids[-1] != eos):
#             input_ids.append(eos)
#             labels.append(eos)

#         return input_ids, labels


# # =============================================================================
# # DataLoader
# # =============================================================================

# class SFTDataLoader:
#     def __init__(self, dataset, batch_size, device, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.device = device
#         self.indices = list(range(len(dataset)))
#         self.pos = 0
#         if shuffle:
#             random.shuffle(self.indices)

#     def get_batch(self):
#         batch_x, batch_y = [], []
#         for _ in range(self.batch_size):
#             if self.pos >= len(self.indices):
#                 self.pos = 0
#             x, y = self.dataset[self.indices[self.pos]]
#             batch_x.append(x)
#             batch_y.append(y)
#             self.pos += 1
#         return (torch.stack(batch_x).to(self.device),
#                 torch.stack(batch_y).to(self.device))


# # =============================================================================
# # 工具函数
# # =============================================================================

# def prepare_data():
#     if os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH):
#         print(f"数据已存在: {TRAIN_PATH}, {VAL_PATH}")
#         return

#     print("准备数据拆分...")
#     with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     total = len(lines)
#     random.seed(42)
#     indices = list(range(total))
#     random.shuffle(indices)

#     val_size = max(50, int(total * VAL_RATIO))
#     val_idx, train_idx = indices[:val_size], indices[val_size:]

#     os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
#     with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
#         for i in train_idx:
#             f.write(lines[i])
#     with open(VAL_PATH, 'w', encoding='utf-8') as f:
#         for i in val_idx:
#             f.write(lines[i])

#     print(f"  训练集: {len(train_idx)} 条, 验证集: {len(val_idx)} 条")


# def get_lr(it):
#     if it < WARMUP_ITERS:
#         return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
#     if it > MAX_ITERS:
#         return MIN_LR
#     ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
#     coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
#     return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# @torch.no_grad()
# def estimate_loss(model, train_loader, val_loader, eval_iters=20):
#     model.eval()
#     out = {}
#     for name, loader in [("train", train_loader), ("val", val_loader)]:
#         losses = []
#         for _ in range(eval_iters):
#             X, Y = loader.get_batch()
#             #  GPT模型接口: model(idx, targets) → (logits, loss)
#             _, loss = model(X, Y)
#             losses.append(loss.item())
#         out[name] = sum(losses) / len(losses)
#     model.train()
#     return out


# # =============================================================================
# # 推理辅助
# # =============================================================================

# @torch.no_grad()
# def chat_inference(model, tokenizer, prompt, max_new_tokens=256,
#                    temperature=0.7, top_k=50, device="cuda"):
#     """用自定义模型做简单对话推理"""
#     model.eval()
#     messages = [{"role": "user", "content": prompt}]
#     input_ids = tokenizer.apply_chat_template(
#         messages, tokenize=True, add_generation_prompt=True, return_tensors=None
#     )
#     idx = torch.tensor([input_ids], dtype=torch.long, device=device)

#     output_ids = model.generate(
#         idx, max_new_tokens=max_new_tokens,
#         temperature=temperature, top_k=top_k,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     # 只取新生成的部分
#     new_ids = output_ids[0, len(input_ids):].tolist()
#     # 截断到 eos
#     if tokenizer.eos_token_id in new_ids:
#         new_ids = new_ids[:new_ids.index(tokenizer.eos_token_id)]
#     return tokenizer.decode(new_ids, skip_special_tokens=True)


# # =============================================================================
# # 主训练流程
# # =============================================================================

# def train():
#     print("=" * 60)
#     print(f"  SFT 微调 (自定义 GPT 模型): {MODEL_NAME}")
#     print(f"  数据: {SFT_DATA_PATH}")
#     print(f"  设备: {DEVICE}, dtype: {DTYPE}")
#     print(f"  LoRA: {'开启' if USE_LORA else '关闭'}")
#     if USE_LORA:
#         print(f"    targets: {LORA_TARGETS}, r={LORA_R}, alpha={LORA_ALPHA}")
#     print("=" * 60)

#     # -------------------- wandb --------------------
#     use_wandb = WANDB_LOG
#     if use_wandb:
#         try:
#             import wandb
#             wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
#                 "model": MODEL_NAME, "use_lora": USE_LORA,
#                 "lora_r": LORA_R if USE_LORA else None,
#                 "lora_alpha": LORA_ALPHA if USE_LORA else None,
#                 "lora_targets": list(LORA_TARGETS) if USE_LORA else None,
#                 "max_seq_len": MAX_SEQ_LEN,
#                 "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
#                 "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
#                 "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
#                 "weight_decay": WEIGHT_DECAY, "dtype": DTYPE,
#             })
#         except Exception as e:
#             print(f"  wandb 初始化失败: {e}")
#             use_wandb = False

#     # 数据
#     prepare_data()

#     print("\n加载 tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print("\n创建数据集...")
#     train_ds = SFTDataset(TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
#     val_ds = SFTDataset(VAL_PATH, tokenizer, MAX_SEQ_LEN)
#     print(f"  训练集: {len(train_ds)} 条, 验证集: {len(val_ds)} 条")

#     train_loader = SFTDataLoader(train_ds, BATCH_SIZE, DEVICE, shuffle=True)
#     val_loader = SFTDataLoader(val_ds, BATCH_SIZE, DEVICE, shuffle=False)


#     # 使用自定义模型加载 HF 权重
#     print(f"\n加载模型: {MODEL_NAME} 加载 GPT")
#     model = GPT.from_pretrained(MODEL_NAME)

#     # 转换精度并移到设备
#     ptdtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
#     model = model.to(dtype=ptdtype, device=DEVICE)

#     # LoRA
#     if USE_LORA:
#         apply_lora(model, r=LORA_R, alpha=LORA_ALPHA,
#                    dropout=LORA_DROPOUT, targets=LORA_TARGETS)
#         mark_only_lora_as_trainable(model)
#         model.to(DEVICE)  # 确保 LoRA 参数也在正确设备

#     model.train()

#     n_params = sum(p.numel() for p in model.parameters())
#     n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  总参数: {n_params/1e6:.1f}M, 可训练: {n_trainable/1e6:.1f}M")

#     if use_wandb:
#         try:
#             import wandb
#             wandb.config.update({
#                 "total_params_M": round(n_params / 1e6, 1),
#                 "trainable_params_M": round(n_trainable / 1e6, 1),
#             })
#         except Exception:
#             pass

#     # 优化器
#     decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
#     nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
#     optimizer = torch.optim.AdamW([
#         {"params": decay_params, "weight_decay": WEIGHT_DECAY},
#         {"params": nodecay_params, "weight_decay": 0.0},
#     ], lr=LEARNING_RATE, betas=(0.9, 0.95))

#     # AMP
#     device_type = "cuda" if "cuda" in DEVICE else "cpu"
#     ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
#     scaler = torch.amp.GradScaler(device_type, enabled=(DTYPE == "float16"))

#     # 训练循环
#     os.makedirs(OUT_DIR, exist_ok=True)
#     best_val_loss = float('inf')
#     patience_cnt = 0
#     iter_num = 0

#     print(f"\n开始训练 (max_iters={MAX_ITERS}, batch={BATCH_SIZE}×{GRAD_ACCUM_STEPS}={BATCH_SIZE*GRAD_ACCUM_STEPS})")
#     print("-" * 60)

#     while iter_num <= MAX_ITERS:
#         lr = get_lr(iter_num)
#         for pg in optimizer.param_groups:
#             pg["lr"] = lr

#         # 评估
#         if iter_num % EVAL_INTERVAL == 0:
#             losses = estimate_loss(model, train_loader, val_loader)
#             train_loss, val_loss = losses["train"], losses["val"]
#             print(f"\n  [eval] step {iter_num}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.2e}")

#             if use_wandb:
#                 try:
#                     import wandb
#                     wandb.log({"eval/train_loss": train_loss, "eval/val_loss": val_loss, "lr": lr}, step=iter_num)
#                 except Exception:
#                     use_wandb = False

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 patience_cnt = 0
#                 print(f"  ✓ 新最优! 保存模型...")
#                 if USE_LORA:
#                     torch.save(lora_state_dict(model), os.path.join(OUT_DIR, "lora_best.pt"))
#                 else:
#                     # 保存自定义模型: config + state_dict
#                     torch.save({
#                         "config": model.config,
#                         "state_dict": model.state_dict(),
#                     }, os.path.join(OUT_DIR, "best_model.pt"))
#                 tokenizer.save_pretrained(OUT_DIR)
#             else:
#                 patience_cnt += 1
#                 print(f"  ✗ 未改善 ({patience_cnt}/{PATIENCE})")
#                 if patience_cnt >= PATIENCE:
#                     print(f"\n  早停触发! best_val_loss={best_val_loss:.4f}")
#                     break

#         # 梯度累积训练
#         optimizer.zero_grad(set_to_none=True)
#         accum_loss = 0.0

#         for micro in range(GRAD_ACCUM_STEPS):
#             X, Y = train_loader.get_batch()
#             with ctx:
#                 # 自定义模型接口
#                 _, loss = model(X, Y)
#                 loss = loss / GRAD_ACCUM_STEPS
#             scaler.scale(loss).backward()
#             accum_loss += loss.item()

#         if GRAD_CLIP > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

#         scaler.step(optimizer)
#         scaler.update()

#         if iter_num % LOG_INTERVAL == 0:
#             total_loss = accum_loss * GRAD_ACCUM_STEPS
#             print(f"  iter {iter_num}: loss={total_loss:.4f}, lr={lr:.2e}")
#             if use_wandb:
#                 try:
#                     import wandb
#                     wandb.log({"train/loss": total_loss, "lr": lr}, step=iter_num)
#                 except Exception:
#                     use_wandb = False

#         iter_num += 1

#     # 保存最终模型
#     print(f"\n保存最终模型到 {OUT_DIR}")
#     if USE_LORA:
#         torch.save(lora_state_dict(model), os.path.join(OUT_DIR, "lora_final.pt"))
#     else:
#         torch.save({
#             "config": model.config,
#             "state_dict": model.state_dict(),
#         }, os.path.join(OUT_DIR, "final_model.pt"))
#     tokenizer.save_pretrained(OUT_DIR)

#     if use_wandb:
#         try:
#             import wandb
#             wandb.run.summary["final_best_val_loss"] = best_val_loss
#             wandb.finish()
#         except Exception:
#             pass

#     print(f"\n{'='*60}")
#     print(f"  训练完成! best_val_loss={best_val_loss:.4f}")
#     print(f"{'='*60}")

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     train()

# 已弃用