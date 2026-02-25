import argparse
import os
import random
import time
import math
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from gpt import GPT
from dataset import SFTDataset, sft_data_generator, IGNORE_TOKEN_ID
from lora import apply_lora, mark_only_lora_as_trainable, lora_state_dict
import checkpoint_manager as ckpt_mgr

# ======================================================================
# 命令行参数
# ======================================================================
parser = argparse.ArgumentParser(description="GPT-CHAT-SFT")
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--data-path", type=str, default="data/sft/sft_data.jsonl")
parser.add_argument("--num-iterations", type=int, default=2000)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--patience", type=int, default=8, help="0表示禁用early-stop")
parser.add_argument("--out-dir", type=str, default="out/sft")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--scratch", action="store_true", help="从头训练")
mode.add_argument("--resume", action="store_true", help="恢复训练")
args = parser.parse_args()


MAX_SEQ_LEN = 1024
GRAD_ACCUM_STEPS = 8
VAL_RATIO = 0.01
LOG_EVERY = 10
LORA_TARGETS = {"c_q", "c_v", "c_k", "c_proj", "lm_head"}
LORA_DROPOUT = 0.1


# ======================================================================
# 工具
# ======================================================================
def prepare_data_split(data_path, out_dir):
    """
    将原始数据拆分为训练集和验证集，
    Returns:
        (train_path, val_path)
    """
    train_path = os.path.join(out_dir, "sft_train.jsonl")
    val_path = os.path.join(out_dir, "sft_val.jsonl")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path

    if not os.path.exists(data_path):
        print(f"源数据文件不存在: {data_path}")
        exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(42)
    indices = list(range(len(lines)))
    random.shuffle(indices)

    val_size = max(50, int(len(lines) * VAL_RATIO))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    os.makedirs(out_dir, exist_ok=True)
    for path, idxs in [(train_path, train_idx), (val_path, val_idx)]:
        with open(path, "w", encoding="utf-8") as f:
            for i in idxs:
                f.write(lines[i])

    print(f"数据拆分: 训练 {len(train_idx)} 条, 验证 {len(val_idx)} 条")
    return train_path, val_path


def get_lr(it, max_iters, warmup_iters, learning_rate):
    min_lr = learning_rate * 0.1
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def evaluate_val_loss(model, val_loader, n_steps, autocast_ctx):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_steps):
            x, y, m = next(val_loader)
            with autocast_ctx:
                _, loss = model(x, y, attn_mask=m)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ======================================================================
# 训练
# ======================================================================

def main():
    # 模式检查
    has_ckpt = ckpt_mgr.exists(args.out_dir)

    assert args.scratch or args.resume or not has_ckpt, \
        f"已发现checkpoint\n请指定--resume继续训练，或--scratch从头开始"

    if args.resume and not has_ckpt:
        print(f"未找到 checkpoint，无法 resume")
        exit(1)

    # resume
    resumed_ckpt = None
    if args.resume:
        resumed_ckpt = ckpt_mgr.load(args.out_dir)
        ckpt_mgr.restore_args(resumed_ckpt, args)
        print(f"从 checkpoint 恢复配置(step={resumed_ckpt['step']})")

    # scratch
    if args.scratch:
        ckpt_mgr.clean(args.out_dir)
        print(f"已清空: {args.out_dir}")

    train_path, val_path = prepare_data_split(args.data_path, os.path.dirname(args.data_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda" else nullcontext()
    )

    eff_batch = args.batch_size * GRAD_ACCUM_STEPS
    lora_alpha = args.lora_r * 2

    # 数据准备
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    train_dataset = SFTDataset(train_path, tokenizer)
    val_dataset = SFTDataset(val_path, tokenizer)

    # 自动计算的训练参数
    steps_per_epoch = max(1, len(train_dataset) // eff_batch)
    eval_every = max(100, steps_per_epoch // 3)
    checkpoint_every = eval_every * 5
    warmup_iters = args.num_iterations // 50
    eval_steps = max(1, len(val_dataset) // (args.batch_size * 10))

    print(f"model: {args.model_name} | device: {device} | batch: {eff_batch} | "
          f"iters: {args.num_iterations} | lr: {args.learning_rate} | "
          f"train: {len(train_dataset)} | val: {len(val_dataset)} | "
          f"eval_every: {eval_every} | LoRA r={args.lora_r}")

    # 数据生成器
    def build_loader(dataset, shuffle=True):
        return sft_data_generator(
            dataset=dataset, tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN, batch_size=args.batch_size,
            device=device, buffer_size=128, shuffle=shuffle,
        )

    train_loader = build_loader(train_dataset, shuffle=True)
    x, y, m = next(train_loader)

    # 模型
    model = GPT.from_pretrained(args.model_name)
    model = model.to(dtype=ptdtype, device=device)

    apply_lora(model, r=args.lora_r, alpha=lora_alpha,
               dropout=LORA_DROPOUT, targets=LORA_TARGETS)
    mark_only_lora_as_trainable(model)
    model.train()

    # 优化器
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler(device_type, enabled=False)

    # 恢复训练状态
    step = 0
    best_val_loss = float("inf")
    patience_cnt = 0

    if resumed_ckpt is not None:
        step, best_val_loss, patience_cnt = ckpt_mgr.restore_state(
            resumed_ckpt, model, optimizer)
        print(f"恢复训练: step={step}, best_val_loss={best_val_loss:.4f}")

    # 训练
    os.makedirs(args.out_dir, exist_ok=True)

    while step < args.num_iterations:
        lr = get_lr(step, args.num_iterations, warmup_iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 评估
        if step % eval_every == 0:
            val_loader = build_loader(val_dataset, shuffle=False)
            val_loss = evaluate_val_loss(model, val_loader, eval_steps, autocast_ctx)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_cnt = 0
                torch.save(lora_state_dict(model),
                           os.path.join(args.out_dir, "lora_best.pt"))
                tokenizer.save_pretrained(args.out_dir)
            else:
                patience_cnt += 1

            mark = "✓" if improved else f"✗ patience {patience_cnt}/{args.patience}"
            print(f"[eval] step {step} | val_loss={val_loss:.4f} | "
                  f"best={best_val_loss:.4f} | {mark}")

            if args.patience > 0 and patience_cnt >= args.patience:
                print(f"早停触发, best_val_loss={best_val_loss:.4f}")
                break

            if step > 0 and step % checkpoint_every == 0:
                ckpt_mgr.save(args.out_dir, step, model, optimizer,
                              best_val_loss, patience_cnt, args)

        # 梯度累积
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            with autocast_ctx:
                _, loss = model(x, y, attn_mask=m)
                loss_scaled = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / GRAD_ACCUM_STEPS
            x, y, m = next(train_loader)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0

        if step % LOG_EVERY == 0:
            print(f"[train] step {step:05d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt*1000:.0f}ms")

        step += 1

    # 保存最终模型
    torch.save(lora_state_dict(model), os.path.join(args.out_dir, "lora_final.pt"))
    ckpt_mgr.save(args.out_dir, step, model, optimizer, best_val_loss, patience_cnt, args)
    print(f"训练完成, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
