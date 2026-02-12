"""
chat_sft.py — 使用 BOS 对齐 + 最佳适配打包的 SFT 训练脚本

适配 GPT 模型框架，架构参考 sample/chat_sft.py

Run as:
    python chat_sft.py

Or with arguments:
    python chat_sft.py --model-name=Qwen/Qwen2.5-0.5B --device-batch-size=4 --grad-accum-steps=8
"""

import argparse
import os
import random
import time
import json
import math
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from gpt import GPT, GPTConfig
from dataset import SFTDataset, sft_data_generator_simple, sft_data_generator_bos_bestfit
from lora import apply_lora, mark_only_lora_as_trainable, lora_state_dict, load_lora_weights

# -----------------------------------------------------------------------------
# 【命令行参数配置】
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) with gradient accumulation")
# 模型
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name or path")
parser.add_argument("--model-tag", type=str, default=None, help="Resume from checkpoint tag")
# 数据
parser.add_argument("--data-path", type=str, default="data/sft/sft_data.jsonl", help="Source data file")
parser.add_argument("--train-path", type=str, default="data/sft/sft_train.jsonl", help="Train split output")
parser.add_argument("--val-path", type=str, default="data/sft/sft_val.jsonl", help="Val split output")
parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation set ratio")
# 训练参数
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
parser.add_argument("--device-batch-size", type=int, default=4, help="Per-device micro batch size")
parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--num-iterations", type=int, default=40000, help="Number of training iterations (optimizer steps)")
parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--warmup-iters", type=int, default=500, help="Warmup iterations")
parser.add_argument("--eval-every", type=int, default=2000, help="Evaluate every N steps")
parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
# 设备
parser.add_argument("--device", type=str, default="cuda", help="Device: cuda|cpu")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16|float16")
# 数据生成器
parser.add_argument("--buffer-size", type=int, default=100, help="Buffer size for bestfit packing")
# LoRA 参数
parser.add_argument("--use-lora", action="store_true",default=True, help="Use LoRA for fine-tuning")
parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
parser.add_argument("--lora-targets", type=str, default="c_q,c_v", help="Comma-separated LoRA target layers")
# 输出
parser.add_argument("--out-dir", type=str, default="out/sft", help="Output directory")
parser.add_argument("--dry-run", action="store_true", help="Skip saving checkpoints")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# 【工具函数】
# -----------------------------------------------------------------------------

def print0(*args, **kwargs):
    print(*args, **kwargs)


def prepare_data_split():
    if os.path.exists(args.train_path) and os.path.exists(args.val_path):
        print0(f"数据已存在: {args.train_path}, {args.val_path}")
        return
    
    if not os.path.exists(args.data_path):
        print0(f"源数据文件不存在: {args.data_path}")
        exit(1)
    
    print0(f"准备数据拆分: {args.data_path}")
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    random.seed(42)
    indices = list(range(total))
    random.shuffle(indices)
    
    val_size = max(50, int(total * args.val_ratio))
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    os.makedirs(os.path.dirname(args.train_path), exist_ok=True)
    
    with open(args.train_path, 'w', encoding='utf-8') as f:
        for i in train_idx:
            f.write(lines[i])
    
    with open(args.val_path, 'w', encoding='utf-8') as f:
        for i in val_idx:
            f.write(lines[i])
    
    print0(f"  训练集: {len(train_idx)} 条, 验证集: {len(val_idx)} 条")


def get_lr(it, max_iters, warmup_iters, learning_rate, min_lr=None):
    if min_lr is None:
        min_lr = learning_rate * 0.1
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# 【主训练流程】
# -----------------------------------------------------------------------------

def main():
    print0("=" * 70)
    print0("SFT 微调 (梯度累积版)")
    print0("=" * 70)
    
    prepare_data_split()
    
    # 设备配置
    device = args.device if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(args.dtype, torch.bfloat16)
    
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # 计算等效 batch size
    eff_batch = args.device_batch_size * args.grad_accum_steps
    total_samples_per_epoch = None  # 下面加载数据集后计算
    
    print0(f"\n配置:")
    print0(f"  模型: {args.model_name}")
    print0(f"  设备: {device}, dtype: {args.dtype}")
    print0(f"  micro_batch_size: {args.device_batch_size}")
    print0(f"  grad_accum_steps: {args.grad_accum_steps}")
    print0(f"  effective_batch_size: {eff_batch}")
    print0(f"  max_seq_len: {args.max_seq_len}")
    print0(f"  num_iterations: {args.num_iterations} (optimizer steps)")
    print0(f"  LoRA: {'启用' if args.use_lora else '禁用'}")
    if args.use_lora:
        print0(f"    r={args.lora_r}, alpha={args.lora_alpha}, targets={args.lora_targets}")
    
    # 加载 tokenizer
    print0(f"\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    print0(f"\n创建数据集...")
    train_dataset = SFTDataset(args.train_path, tokenizer)
    val_dataset = SFTDataset(args.val_path, tokenizer)
    print0(f"  训练集: {len(train_dataset)} 条对话")
    print0(f"  验证集: {len(val_dataset)} 条对话")
    
    # 计算 epoch 信息
    steps_per_epoch = len(train_dataset) // eff_batch
    total_epochs = args.num_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
    print0(f"  steps_per_epoch: {steps_per_epoch}")
    print0(f"  total_epochs: {total_epochs:.2f}")
    
    # 统计长度分布
    sample_size = min(1000, len(train_dataset))
    lengths = [len(train_dataset.get_conversation_tokens(i)) for i in range(sample_size)]
    print0(f"  对话长度分布 (前{sample_size}条): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    
    # 创建数据生成器（bestfit packing，~2x 效率提升）
    print0(f"\n创建数据生成器 (bestfit packing, buffer_size={args.buffer_size})...")
    
    train_loader = sft_data_generator_bos_bestfit(
        train_dataset, tokenizer, args.max_seq_len, args.device_batch_size, device, args.buffer_size
    )
    
    def build_val_loader():
        return sft_data_generator_bos_bestfit(
            val_dataset, tokenizer, args.max_seq_len, args.device_batch_size, device, args.buffer_size
        )
    
    # 加载模型
    print0(f"\n加载模型...")
    model = GPT.from_pretrained(args.model_name)
    model = model.to(dtype=ptdtype, device=device)
    
    # 注入 LoRA
    if args.use_lora:
        lora_targets = set(args.lora_targets.split(","))
        apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, 
                   dropout=args.lora_dropout, targets=lora_targets)
        mark_only_lora_as_trainable(model)
    
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"  总参数: {n_params/1e6:.1f}M")
    print0(f"  可训练参数: {n_trainable/1e6:.1f}M ({100*n_trainable/n_params:.2f}%)")
    
    # 优化器
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95))
    
    # AMP
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))
    
    # 训练状态
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_cnt = 0
    step = 0  # optimizer step 计数
    
    print0(f"\n开始训练 (max_iters={args.num_iterations}, grad_accum={args.grad_accum_steps})")
    print0("-" * 70)
    
    while step < args.num_iterations:
        # 学习率
        lr = get_lr(step, args.num_iterations, args.warmup_iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # 定期评估
        if step % args.eval_every == 0:
            model.eval()
            val_loader = build_val_loader()
            val_losses = []
            eval_iters = min(20, len(val_dataset) // args.device_batch_size)
            with torch.no_grad():
                for _ in range(eval_iters):
                    X_val, Y_val = next(val_loader)
                    with autocast_ctx:
                        _, loss_val = model(X_val, Y_val)
                    val_losses.append(loss_val.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()
            
            print0(f"\n[eval] step {step}: val_loss={val_loss:.4f}, lr={lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                print0(f"  ✓ 新最优! 保存模型...")
                
                if not args.dry_run:
                    if args.use_lora:
                        torch.save(lora_state_dict(model), os.path.join(args.out_dir, "lora_best.pt"))
                    else:
                        torch.save({
                            "config": model.config,
                            "state_dict": model.state_dict(),
                            "step": step,
                            "val_loss": val_loss,
                            "user_config": user_config,
                        }, os.path.join(args.out_dir, "best_model.pt"))
                    tokenizer.save_pretrained(args.out_dir)
            else:
                patience_cnt += 1
                print0(f"  ✗ 未改善 ({patience_cnt}/{args.patience})")
                if patience_cnt >= args.patience:
                    print0(f"\n早停触发! best_val_loss={best_val_loss:.4f}")
                    break
        
        # ====================================================================
        # 【梯度累积训练步骤】
        # ====================================================================
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        accum_loss = 0.0
        
        for micro_step in range(args.grad_accum_steps):
            x, y = next(train_loader)
            
            # 最后一个 micro step 正常同步，其余不同步梯度（单卡无影响，多卡 DDP 需要）
            with autocast_ctx:
                _, loss = model(x, y)
                # 除以 grad_accum_steps 使梯度等效于大 batch
                loss = loss / args.grad_accum_steps
            
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        
        # 梯度裁剪 + 优化器更新
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        dt = time.time() - t0
        
        # 日志（accum_loss 已经是平均值）
        if step % args.log_every == 0:
            # 计算 tokens/sec
            toks_per_step = eff_batch * args.max_seq_len
            toks_per_sec = toks_per_step / dt
            print0(f"step {step:05d} | loss: {accum_loss:.4f} | lr: {lr:.2e} | "
                   f"dt: {dt*1000:.0f}ms | tok/s: {toks_per_sec:.0f}")
        
        step += 1
    
    # 保存最终模型
    print0(f"\n保存最终模型...")
    if not args.dry_run:
        if args.use_lora:
            torch.save(lora_state_dict(model), os.path.join(args.out_dir, "lora_final.pt"))
        else:
            torch.save({
                "config": model.config,
                "state_dict": model.state_dict(),
                "step": step,
                "val_loss": val_loss if 'val_loss' in locals() else None,
                "user_config": user_config,
            }, os.path.join(args.out_dir, "final_model.pt"))
    
    print0(f"\n{'='*70}")
    print0(f"训练完成! best_val_loss={best_val_loss:.4f}")
    print0(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()