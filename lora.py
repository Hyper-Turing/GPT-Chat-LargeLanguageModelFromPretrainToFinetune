"""
lora.py - 从零实现 LoRA，适配自定义 Qwen2.5 模型

原理：
  原始层:  y = Wx
  LoRA层:  y = Wx + (α/r) · BAx
  
  W: (out, in) 冻结不训练
  A: (r, in)   随机初始化，可训练
  B: (out, r)  零初始化，可训练
  
  r=16 时，1.5B 模型只需训练 ~4M 参数（原来的 0.3%）
  显存从 ~12GB 降到 ~4GB

用法：
  from lora import apply_lora, lora_state_dict, mark_only_lora_as_trainable

  model = GPT.from_pretrained("Qwen/Qwen2.5-1.5B")
  apply_lora(model, r=16, alpha=32, targets={"c_q", "c_v"})
  mark_only_lora_as_trainable(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================================
# 核心：LoRA Linear 层
# =====================================================================

class LoRALinear(nn.Module):
    """
    替换 nn.Linear，添加低秩旁路
    
    forward:
      y = F.linear(x, W, bias) + (alpha/r) * x @ A^T @ B^T
      
    参数量对比（以 c_q 为例，Qwen2.5-1.5B: in=1536, out=1536）：
      原始 W: 1536 × 1536 = 2,359,296
      LoRA A: 16 × 1536   = 24,576
      LoRA B: 1536 × 16   = 24,576
      LoRA 总: 49,152（原来的 2%）
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

        # ✅ 获取原始权重的设备
        device = original_linear.weight.device

        # ✅ 关键修复：LoRA 参数始终使用 float32，避免 bfloat16 精度损失
        # 基座模型可以是 bfloat16（节省显存），但 LoRA 参数用 float32（保证精度）
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32, device=device))  # B 零初始化

        # A 用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 可选 dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _apply(self, fn):
        """重写 _apply 方法，防止 LoRA 参数被转换为 float16/bfloat16"""
        # 先保存 LoRA 参数的原始 dtype
        lora_A_dtype = self.lora_A.dtype
        lora_B_dtype = self.lora_B.dtype

        # 调用父类的 _apply（会转换所有参数）
        super()._apply(fn)

        # 强制将 LoRA 参数转回 float32
        if self.lora_A.dtype != lora_A_dtype:
            self.lora_A.data = self.lora_A.data.to(lora_A_dtype)
        if self.lora_B.dtype != lora_B_dtype:
            self.lora_B.data = self.lora_B.data.to(lora_B_dtype)

        return self

    def forward(self, x):
        # 原始路径（冻结权重）
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA 旁路: x @ A^T @ B^T * scaling
        # ✅ 关键修复：将输入转为 float32 进行 LoRA 计算，保证精度
        x_lora = self.lora_dropout(x)

        # 如果输入不是 float32，转换为 float32 进行高精度计算
        if x_lora.dtype != torch.float32:
            x_lora = x_lora.to(torch.float32)
            lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)
            # 转回原始 dtype
            lora_out = lora_out.to(base_out.dtype)
        else:
            lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)

        return base_out + lora_out * self.scaling
    
    def merge_weights(self):
        """推理时将 LoRA 合并回原始权重，消除额外计算开销"""
        with torch.no_grad():
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        # 合并后可以删掉 LoRA 参数
        return nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)


# =====================================================================
# 工具函数：给模型注入 LoRA
# =====================================================================

def apply_lora(model, r=16, alpha=32.0, dropout=0.0,
               targets=None):
    """
    遍历模型，将指定的 nn.Linear 替换为 LoRALinear
    
    Args:
        model: GPT 模型实例
        r: LoRA 秩（推荐 8~64，越大越强但越慢）
        alpha: 缩放系数（通常 alpha = 2*r）
        dropout: LoRA dropout
        targets: 要注入 LoRA 的层名集合，默认 Q 和 V
                 常见选择：
                   {"c_q", "c_v"}           — 最小，效果不错
                   {"c_q", "c_k", "c_v"}    — 加上 K
                   {"c_q", "c_v", "c_proj"}  — 加上输出投影
                   {"c_q", "c_v", "gate_proj", "up_proj", "down_proj"}  — 加上 MLP
    
    Returns:
        lora_params: 所有 LoRA 参数列表（方便传给优化器）
    """
    if targets is None:
        targets = {"c_q", "c_v"}  # 默认只对 Q 和 V 注入
    
    count = 0
    for name, module in model.named_modules():
        # 遍历当前模块的直接子模块
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr_name in targets:
                # 替换为 LoRA 版本
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, attr_name, lora_layer)
                count += 1
    
    print(f"LoRA 注入完成: {count} 个层, r={r}, alpha={alpha}, targets={targets}")
    return count


def mark_only_lora_as_trainable(model):
    """
    冻结所有参数，只保留 LoRA 参数可训练
    """
    # 先冻结全部
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻 LoRA 参数
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
    """只提取 LoRA 参数（保存时用，文件极小）"""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_weights(model, path):
    """加载 LoRA 权重（只加载 lora_ 参数，其余保持不变）"""
    lora_sd = torch.load(path, map_location="cpu")
    model.load_state_dict(lora_sd, strict=False)
    print(f"加载 LoRA 权重: {len(lora_sd)} 个参数, 来自 {path}")


def merge_and_save(model, save_path):
    """
    将 LoRA 合并回基础权重，保存为完整模型
    合并后推理速度与原始模型一样（无额外开销）
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.weight.data += module.scaling * (module.lora_B @ module.lora_A)
    
    # 保存完整 state_dict
    torch.save(model.state_dict(), save_path)
    print(f"合并后完整模型已保存: {save_path}")