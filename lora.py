"""
从零实现LoRA

原理
原始层: y = Wx
LoRA层: y = Wx + (a/r) · BAx

W: (out, in) 原始权重, 冻结不训练
A: (r, in)   随机初始化, 训练
B: (out, r)  零初始化, 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    替换Linear 添加低秩旁路
    forward:
        y = F.linear(x, W, bias) + (alpha/r) * x @ A^T @ B^T
    """
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.scaling = alpha / r  # LoRA 缩放因子

        # 保留原始权重（冻结）
        self.weight = original_linear.weight  # 不复制，直接引用
        self.bias = original_linear.bias

        device = original_linear.weight.device

        # 混合精度 避免 bfloat16 精度损失
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=torch.float32, device=device))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始路径 Wx
        base_out = F.linear(x, self.weight, self.bias)
        x_lora = self.lora_dropout(x)

        # LoRA 旁路: x @ A^T @ B^T * scaling
        lora_A = self.lora_A.float()
        lora_B = self.lora_B.float()
        x_lora = x_lora.float()
        lora_out = F.linear(F.linear(x_lora, lora_A), lora_B)
        # 转回原始 dtype
        lora_out = lora_out.to(base_out.dtype)

        return base_out + lora_out * self.scaling
    
# 注入LoRA
def apply_lora(model, r=16, alpha=32.0, dropout=0.0, targets=None):
    """
    遍历模型，将指定的 nn.Linear 替换为 LoRALinear
    
    Args:
        model: GPT 模型实例
        r: LoRA 秩（推荐 8~64，越大越强但越慢）
        alpha: 缩放系数（通常 alpha = 2*r）
        targets: 要注入 LoRA 的层名集合
                {c_q, c_v, c_k, c_proj, lm_head}
    """
    if targets is None:
        targets = {"c_q", "c_v"}  # 默认只对 Q 和 V 注入

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr_name in targets:
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, attr_name, lora_layer)
                count += 1

    print(f"LoRA 注入完成: {count} 个层, r={r}, alpha={alpha}, targets={targets}")
    return count

def mark_only_lora_as_trainable(model):
    # 只保留LoRA可训练
    # 先冻结所有权重
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 LoRA
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable += param.numel()
    
    pct = 100.0 * trainable / total
    print(f"可训练参数: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable

def lora_state_dict(model):
    # 只提取 LoRA 参数
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}

def load_lora_weights(model, path):
    lora_sd = torch.load(path, map_location="cpu")
    model.load_state_dict(lora_sd, strict=False)
    print(f"加载 LoRA 权重: {len(lora_sd)} 个参数, 来自 {path}")

def merge_lora_weights(model, save_path):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.weight.data += module.scaling * (module.lora_B @ module.lora_A)
    
    # 保存完整 state_dict
    torch.save(model.state_dict(), save_path)
    print(f"合并后完整模型已保存: {save_path}")