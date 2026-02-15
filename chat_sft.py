"""
chat_sft.py — SFT 微调训练脚本（重写版）

核心改动:
  1. 使用重写的 dataset.py (基于 apply_chat_template)
  2. 简化训练循环，去除冗余逻辑
  3. 保留 gpt.py + lora.py 核心架构

Run:
    python chat_sft.py --model-name=Qwen/Qwen2.5-0.5B --device-batch-size=4
"""

import argparse
import os
import random
import time
import math
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from gpt import GPT
from dataset import SFTDataset, sft_data_generator
from lora import apply_lora, mark_only_lora_as_trainable, lora_state_dict

# -----------------------------------------------------------------------------
# 命令行参数
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SFT fine-tuning")
# 模型
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
# 数据
parser.add_argument("--data-path", type=str, default="data/sft/sft_data.jsonl")
parser.add_argument("--train-path", type=str, default="data/sft/sft_train.jsonl")
parser.add_argument("--val-path", type=str, default="data/sft/sft_val.jsonl")
parser.add_argument("--val-ratio", type=float, default=0.01)
# 训练
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--grad-accum-steps", type=int, default=8)
parser.add_argument("--num-iterations", type=int, default=40000)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--warmup-iters", type=int, default=500)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--patience", type=int, default=8)
parser.add_argument("--eval-iters", type=int, default=20)
parser.add_argument("--min-steps", type=int, default=0)
parser.add_argument("--disable-early-stop", action="store_true")
# 设备
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="bfloat16")
# LoRA
parser.add_argument("--use-lora", action="store_true", default=True)
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.1)
parser.add_argument("--lora-targets", type=str, default="c_q,c_v,c_k,c_proj")
# 输出
parser.add_argument("--out-dir", type=str, default="out/sft")
parser.add_argument("--checkpoint-every", type=int, default=2000)
# 数据量限制（调试用）
parser.add_argument("--max-train-samples", type=int, default=None)
parser.add_argument("--max-val-samples", type=int, default=None)
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def prepare_data_split():
    """将原始数据拆分为训练集和验证集"""
    if os.path.exists(args.train_path) and os.path.exists(args.val_path):
        print(f"数据已存在: {args.train_path}, {args.val_path}")
        return

    if not os.path.exists(args.data_path):
        print(f"源数据文件不存在: {args.data_path}")
        exit(1)

    with open(args.data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    random.seed(42)
    indices = list(range(total))
    random.shuffle(indices)

    val_size = max(50, int(total * args.val_ratio))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    os.makedirs(os.path.dirname(args.train_path), exist_ok=True)

    with open(args.train_path, "w", encoding="utf-8") as f:
        for i in train_idx:
            f.write(lines[i])

    with open(args.val_path, "w", encoding="utf-8") as f:
        for i in val_idx:
            f.write(lines[i])

    print(f"  训练集: {len(train_idx)} 条, 验证集: {len(val_idx)} 条")


def get_lr(it, max_iters, warmup_iters, learning_rate, min_lr=None):
    """Cosine decay with warmup"""
    if min_lr is None:
        min_lr = learning_rate * 0.1
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class _SubsetDataset:
    """数据集子集包装器"""
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def get_conversation_tokens(self, idx):
        return self.base.get_conversation_tokens(self.indices[idx])

    def get_conversation_tokens_with_mask(self, idx):
        return self.base.get_conversation_tokens_with_mask(self.indices[idx])


def maybe_subset(dataset, max_samples, seed=42):
    if not max_samples or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    return _SubsetDataset(dataset, indices[:max_samples])


# -----------------------------------------------------------------------------
# 主训练流程
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SFT 微调（重写版）")
    print("=" * 70)

    prepare_data_split()

    # 设备
    device = args.device if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype, torch.bfloat16)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    eff_batch = args.device_batch_size * args.grad_accum_steps

    print(f"\n配置:")
    print(f"  模型: {args.model_name}")
    print(f"  设备: {device}, dtype: {args.dtype}")
    print(f"  effective_batch_size: {eff_batch}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  num_iterations: {args.num_iterations}")
    if args.use_lora:
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, targets={args.lora_targets}")

    # Tokenizer
    print(f"\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # 不设置 pad_token = eos_token，dataset.py 内部处理
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  <|im_end|> id: {im_end_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")

    # 数据集
    print(f"\n创建数据集...")
    train_dataset = SFTDataset(args.train_path, tokenizer)
    val_dataset = SFTDataset(args.val_path, tokenizer)

    train_dataset = maybe_subset(train_dataset, args.max_train_samples)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples)
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")

    steps_per_epoch = len(train_dataset) // eff_batch
    total_epochs = args.num_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"  steps_per_epoch: {steps_per_epoch}")
    print(f"  total_epochs: {total_epochs:.2f}")

    # 数据长度分布
    sample_size = min(500, len(train_dataset))
    lengths = [len(train_dataset.get_conversation_tokens(i)) for i in range(sample_size)]
    print(f"  对话长度 (前{sample_size}条): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

    # 数据生成器
    train_loader = sft_data_generator(train_dataset, tokenizer, args.max_seq_len, args.device_batch_size, device)

    def build_val_loader():
        return sft_data_generator(val_dataset, tokenizer, args.max_seq_len, args.device_batch_size, device)

    # ================================================================
    # 模型初始化
    # ================================================================
    print(f"\n加载模型: {args.model_name}")
    model = GPT.from_pretrained(args.model_name)
    model = model.to(dtype=ptdtype, device=device)

    if args.use_lora:
        lora_targets = set(args.lora_targets.split(","))
        apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                   dropout=args.lora_dropout, targets=lora_targets)
        mark_only_lora_as_trainable(model)

    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {n_params/1e6:.1f}M, 可训练: {n_trainable/1e6:.1f}M ({100*n_trainable/n_params:.2f}%)")

    # 优化器
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))

    # ================================================================
    # 训练循环
    # ================================================================
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_cnt = 0
    step = 0

    print(f"\n开始训练 (max_iters={args.num_iterations})")
    print("-" * 70)

    while step < args.num_iterations:
        lr = get_lr(step, args.num_iterations, args.warmup_iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 评估
        if step % args.eval_every == 0:
            model.eval()
            val_loader = build_val_loader()
            val_losses = []
            eval_iters = min(args.eval_iters, max(1, len(val_dataset) // args.device_batch_size))
            with torch.no_grad():
                for _ in range(eval_iters):
                    X_val, Y_val, M_val = next(val_loader)
                    with autocast_ctx:
                        _, loss_val = model(X_val, Y_val, attn_mask=M_val)
                    val_losses.append(loss_val.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()

            print(f"\n[eval] step {step}: val_loss={val_loss:.4f}, lr={lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                print(f"  ✓ 新最优! 保存模型...")
                if args.use_lora:
                    torch.save(lora_state_dict(model), os.path.join(args.out_dir, "lora_best.pt"))
                else:
                    torch.save({"config": model.config, "state_dict": model.state_dict(),
                                "step": step, "val_loss": val_loss},
                               os.path.join(args.out_dir, "best_model.pt"))
                tokenizer.save_pretrained(args.out_dir)
            else:
                patience_cnt += 1
                print(f"  ✗ 未改善 ({patience_cnt}/{args.patience})")
                if not args.disable_early_stop and step >= args.min_steps and patience_cnt >= args.patience:
                    print(f"\n早停触发! best_val_loss={best_val_loss:.4f}")
                    break

            # 定期 checkpoint
            if step % args.checkpoint_every == 0 and step > 0:
                ckpt_path = os.path.join(args.out_dir, f"checkpoint_{step:06d}.pt")
                if args.use_lora:
                    torch.save(lora_state_dict(model), ckpt_path)
                print(f"  保存 checkpoint: {ckpt_path}")

        # 梯度累积训练
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            x, y, m = next(train_loader)
            with autocast_ctx:
                _, loss = model(x, y, attn_mask=m)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0

        if step % args.log_every == 0:
            toks_per_sec = eff_batch * args.max_seq_len / dt
            print(f"step {step:05d} | loss: {accum_loss:.4f} | lr: {lr:.2e} | "
                  f"dt: {dt*1000:.0f}ms | tok/s: {toks_per_sec:.0f}")

        step += 1

    # 保存最终模型
    print(f"\n保存最终模型...")
    if args.use_lora:
        torch.save(lora_state_dict(model), os.path.join(args.out_dir, "lora_final.pt"))

    print(f"\n{'='*70}")
    print(f"训练完成! best_val_loss={best_val_loss:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()