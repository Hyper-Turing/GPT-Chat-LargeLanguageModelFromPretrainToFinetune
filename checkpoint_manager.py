import os
import shutil

import torch

from lora import lora_state_dict

CHECKPOINT_FILE = "checkpoint.pt"


def save(out_dir, step, model, optimizer, best_val_loss, patience_cnt, train_args):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "step": step,
        "lora_state_dict": lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_cnt": patience_cnt,
        "args": vars(train_args),
    }, os.path.join(out_dir, CHECKPOINT_FILE))


def load(out_dir, device="cpu"):
    path = os.path.join(out_dir, CHECKPOINT_FILE)
    return torch.load(path, map_location=device)


def exists(out_dir):
    return os.path.exists(os.path.join(out_dir, CHECKPOINT_FILE))


def clean(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)


def restore_args(ckpt, args):
    """
    恢复训练配置
    """
    saved = ckpt["args"]
    for key in ["model_name", "data_path", "num_iterations", "learning_rate",
                 "batch_size", "lora_r", "patience"]:
        if key in saved:
            setattr(args, key, saved[key])


def restore_state(ckpt, model, optimizer):
    """
    从 checkpoint 恢复模型和优化器状态
    Returns:
        (step, best_val_loss, patience_cnt)
    """
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        ckpt["step"],
        ckpt["best_val_loss"],
        ckpt.get("patience_cnt", 0),
    )
