"""
Checkpoint 管理器 - 支持保存和加载训练状态

功能：
1. 保存 checkpoint（model + optimizer + training state）
2. 加载 checkpoint 恢复训练
3. 自动查找最新的 checkpoint
"""

import os
import glob
import json
import torch


def save_checkpoint(out_dir, step, model, optimizer, best_val_loss, user_config=None):
    """
    保存训练 checkpoint
    
    Args:
        out_dir: 输出目录
        step: 当前训练步数
        model: 模型
        optimizer: 优化器
        best_val_loss: 最优验证损失
        user_config: 用户配置（可选）
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存模型和训练状态
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "user_config": user_config,
    }
    
    checkpoint_path = os.path.join(out_dir, f"checkpoint_{step:06d}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"  保存 checkpoint: {checkpoint_path}")
    
    # 同时保存为 latest.pt 方便恢复
    latest_path = os.path.join(out_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    # 保存元数据（JSON 格式，方便查看）
    meta = {
        "step": step,
        "best_val_loss": best_val_loss,
        "checkpoint_file": f"checkpoint_{step:06d}.pt",
    }
    meta_path = os.path.join(out_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    # 同时更新 latest.json
    latest_meta_path = os.path.join(out_dir, "latest.json")
    with open(latest_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """
    加载 checkpoint
    
    Args:
        checkpoint_path: checkpoint 文件路径
        model: 模型（用于加载状态）
        optimizer: 优化器（可选，用于加载状态）
        device: 设备
    
    Returns:
        step, best_val_loss: 恢复的训练状态
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"  加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 加载优化器状态（如果提供了优化器）
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"  优化器状态已恢复")
    
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    
    print(f"  恢复训练: step={step}, best_val_loss={best_val_loss:.4f}")
    
    return step, best_val_loss


def find_latest_checkpoint(out_dir):
    """
    在目录中查找最新的 checkpoint
    
    Args:
        out_dir: 输出目录
    
    Returns:
        checkpoint_path: 最新 checkpoint 的路径，如果没有则返回 None
    """
    # 首先检查 latest.pt
    latest_path = os.path.join(out_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # 如果没有 latest.pt，查找 checkpoint_*.pt 中步数最大的
    checkpoint_files = glob.glob(os.path.join(out_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        return None
    
    # 按步数排序
    def get_step(filename):
        basename = os.path.basename(filename)
        step_str = basename.replace("checkpoint_", "").replace(".pt", "")
        return int(step_str)
    
    checkpoint_files.sort(key=get_step, reverse=True)
    return checkpoint_files[0]


def resume_from_checkpoint(out_dir, model, optimizer=None, device="cuda"):
    """
    从最新的 checkpoint 恢复训练
    
    Args:
        out_dir: 输出目录
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
    
    Returns:
        step, best_val_loss: 恢复的训练状态，如果没有 checkpoint 则返回 (0, inf)
    """
    checkpoint_path = find_latest_checkpoint(out_dir)
    
    if checkpoint_path is None:
        print(f"  未找到 checkpoint，从头开始训练")
        return 0, float('inf')
    
    return load_checkpoint(checkpoint_path, model, optimizer, device)