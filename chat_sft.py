"""
chat_sft.py — 使用 BOS 对齐 + 最佳适配打包的 SFT 训练脚本

适配 GPT 模型框架，架构参考 sample/chat_sft.py

Run as:
    python chat_sft.py

Or with arguments:
    python chat_sft.py --model-name=Qwen/Qwen2.5-0.5B --device-batch-size=4
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
from dataset import SFTDataset, sft_data_generator_bos_bestfit
from checkpoint_manager import save_checkpoint, resume_from_checkpoint


# -----------------------------------------------------------------------------
# 【命令行参数配置】
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) with BOS-aligned bestfit packing")
# 模型
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
parser.add_argument("--model-tag", type=str, default=None, help="Resume from checkpoint tag")
# 数据
parser.add_argument("--data-path", type=str, default="data/sft/sft_data.jsonl", help="Source data file")
parser.add_argument("--train-path", type=str, default="data/sft/sft_train.jsonl", help="Train split output")
parser.add_argument("--val-path", type=str, default="data/sft/sft_val.jsonl", help="Val split output")
parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation set ratio")
# 训练参数
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
parser.add_argument("--device-batch-size", type=int, default=2, help="Per-device batch size")
parser.add_argument("--num-iterations", type=int, default=3000, help="Number of training iterations")
parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--warmup-iters", type=int, default=100, help="Warmup iterations")
parser.add_argument("--eval-every", type=int, default=200, help="Evaluate every N steps")
parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
# 设备
parser.add_argument("--device", type=str, default="cuda", help="Device: cuda|cpu")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16|float16")
# 数据生成器
parser.add_argument("--buffer-size", type=int, default=100, help="Buffer size for bestfit packing")
# 输出
parser.add_argument("--out-dir", type=str, default="out/sft_bestfit", help="Output directory")
parser.add_argument("--dry-run", action="store_true", help="Skip saving checkpoints")
# wandb
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--wandb-project", type=str, default="qwen-sft", help="Wandb project name")
parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
# checkpoint
parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: out-dir)")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# 【工具函数】
# -----------------------------------------------------------------------------

def print0(*args, **kwargs):
    """只在主进程打印（简化版，单进程直接打印）"""
    print(*args, **kwargs)


def prepare_data_split():
    """
    从 sft_data.jsonl 拆分训练集和验证集
    如果没有现成的 train/val 文件，则自动拆分
    """
    if os.path.exists(args.train_path) and os.path.exists(args.val_path):
        print0(f"数据已存在: {args.train_path}, {args.val_path}")
        return
    
    if not os.path.exists(args.data_path):
        print0(f"源数据文件不存在: {args.data_path}")
        print0("请提供有效的数据文件路径")
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


def get_lr(it, max_iters, warmup_iters, learning_rate, min_lr=0.0):
    """学习率调度：warmup + cosine decay"""
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
    print0("SFT 微调 (BOS 对齐 + 最佳适配打包)")
    print0("=" * 70)
    
    # 准备数据
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
    
    print0(f"\n配置:")
    print0(f"  模型: {args.model_name}")
    print0(f"  设备: {device}, dtype: {args.dtype}")
    print0(f"  batch_size: {args.device_batch_size}")
    print0(f"  max_seq_len: {args.max_seq_len}")
    
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
    
    # 统计长度分布
    sample_size = min(1000, len(train_dataset))
    lengths = [len(train_dataset.get_conversation_tokens(i)) for i in range(sample_size)]
    print0(f"  对话长度分布 (前{sample_size}条): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    
    # 创建数据生成器
    print0(f"\n创建数据生成器 (buffer_size={args.buffer_size})...")
    
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
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"  总参数: {n_params/1e6:.1f}M")
    
    # 优化器
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95))
    
    # AMP
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))
    
    # 确定 checkpoint 目录
    checkpoint_dir = args.checkpoint_dir or args.out_dir
    
    # 尝试恢复 checkpoint
    start_step = 0
    best_val_loss = float('inf')
    if args.resume:
        print0(f"\n尝试从 checkpoint 恢复...")
        start_step, best_val_loss = resume_from_checkpoint(
            checkpoint_dir, model, optimizer, device
        )
    
    # 初始化 wandb
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"sft-{args.model_name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "max_seq_len": args.max_seq_len,
                    "batch_size": args.device_batch_size,
                    "num_iterations": args.num_iterations,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dtype": args.dtype,
                }
            )
            print0(f"\nwandb 已启用: project={args.wandb_project}, run={run_name}")
        except Exception as e:
            print0(f"\nwandb 初始化失败: {e}")
            use_wandb = False
    
    # 训练状态
    os.makedirs(args.out_dir, exist_ok=True)
    patience_cnt = 0
    step = start_step
    
    # 预取第一个 batch（如果从头开始）
    if step == 0:
        x, y = next(train_loader)
    else:
        # 从恢复的 step 开始，需要跳过前面已经训练过的数据
        # 简单处理：继续从数据生成器取数据
        x, y = next(train_loader)
    
    print0(f"\n开始训练 (max_iters={args.num_iterations}, start_step={step})")
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
            eval_iters = 20
            with torch.no_grad():
                for _ in range(eval_iters):
                    X_val, Y_val = next(val_loader)
                    with autocast_ctx:
                        _, loss_val = model(X_val, Y_val)
                    val_losses.append(loss_val.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()
            
            print0(f"\n[eval] step {step}: val_loss={val_loss:.4f}, lr={lr:.2e}")
            
            # wandb 日志
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "step": step,
                        "val/loss": val_loss,
                        "train/lr": lr,
                    }, step=step)
                except Exception:
                    pass
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                print0(f"  ✓ 新最优! 保存模型...")
                
                if not args.dry_run:
                    # 保存模型
                    torch.save({
                        "config": model.config,
                        "state_dict": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "user_config": user_config,
                    }, os.path.join(args.out_dir, "best_model.pt"))
                    tokenizer.save_pretrained(args.out_dir)
                    
                    # 保存 checkpoint（用于断点续训）
                    save_checkpoint(
                        checkpoint_dir, step, model, optimizer, 
                        best_val_loss, user_config
                    )
            else:
                patience_cnt += 1
                print0(f"  ✗ 未改善 ({patience_cnt}/{args.patience})")
                if patience_cnt >= args.patience:
                    print0(f"\n早停触发! best_val_loss={best_val_loss:.4f}")
                    break
        
        # 训练步骤
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        
        _, loss = model(x, y)
        loss.backward()
        
        if args.weight_decay > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        dt = time.time() - t0
        
        # 预取下一个 batch
        x, y = next(train_loader)
        
        # 日志
        if step % args.log_every == 0:
            print0(f"step {step:05d} | loss: {loss.item():.6f} | lr: {lr:.2e} | dt: {dt*1000:.2f}ms")
            
            # wandb 日志
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/dt": dt * 1000,
                    }, step=step)
                except Exception:
                    pass
        
        step += 1
    
    # 保存最终模型
    print0(f"\n保存最终模型...")
    if not args.dry_run:
        torch.save({
            "config": model.config,
            "state_dict": model.state_dict(),
            "step": step,
            "val_loss": val_loss if 'val_loss' in locals() else None,
            "user_config": user_config,
        }, os.path.join(args.out_dir, "final_model.pt"))
        
        # 保存最终 checkpoint
        save_checkpoint(
            checkpoint_dir, step, model, optimizer,
            best_val_loss, user_config
        )
    
    # 结束 wandb
    if use_wandb:
        try:
            import wandb
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.finish()
        except Exception:
            pass
    
    print0(f"\n{'='*70}")
    print0(f"训练完成! best_val_loss={best_val_loss:.4f}")
    print0(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()