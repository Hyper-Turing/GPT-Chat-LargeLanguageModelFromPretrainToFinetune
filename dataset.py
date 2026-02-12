"""
SFT 数据生成器 - BOS 对齐 + 最佳适配打包 + Qwen Label Mask

对齐 Qwen 官方 mask 策略：
1. <|im_start|>, <|im_end|> 等特殊 token 不 mask（计算 loss）
2. system 和每轮 user 的内容添加 mask（不计算 loss）
3. 每轮对话中的角色信息（"system\\n", "user\\n", "assistant\\n"）添加 mask
4. assistant 的回复内容计算 loss
5. padding 位置 mask 为 -100

即：
  <|im_start|> → 算 loss
  system\\n     → 不算
  内容          → 不算
  <|im_end|>   → 算 loss
  \\n           → 不算

  <|im_start|> → 算 loss
  user\\n       → 不算
  内容          → 不算
  <|im_end|>   → 算 loss
  \\n           → 不算

  <|im_start|> → 算 loss
  assistant\\n  → 不算
  内容          → 算 loss
  <|im_end|>   → 算 loss
  \\n           → 不算
"""

import os
import json
import torch
from transformers import AutoTokenizer


class SFTDataset:
    """
    SFT 对话数据集，从 JSONL 加载对话数据
    
    支持的数据格式:
      - {"conversations": [{"role": "user", "content": "..."}, ...]}
      - {"messages": [{"role": "user", "content": "..."}, ...]}
      - {"instruction": "...", "input": "...", "output": "..."}
    """
    
    def __init__(self, filepath: str, tokenizer: AutoTokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        
        # 预编码角色标记 token ids，用于精确定位
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")  # 151644
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")      # 151645
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)        # [198]
        
        # 预编码角色名 + \n 的 token ids
        self.role_tokens = {}
        for role in ["system", "user", "assistant"]:
            # "system\n" / "user\n" / "assistant\n"
            ids = tokenizer.encode(f"{role}\n", add_special_tokens=False)
            self.role_tokens[role] = ids
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_conversation_tokens(self, idx: int) -> list[int]:
        """获取指定对话的完整 token IDs"""
        tokens, _ = self.get_conversation_tokens_with_mask(idx)
        return tokens
    
    def get_conversation_tokens_with_mask(self, idx: int) -> tuple[list[int], list[int]]:
        """
        获取对话 token IDs 和 loss mask
        
        Mask 策略：
        - <|im_start|>: loss_mask = 1（计算 loss）
        - "role\\n": loss_mask = 0（不计算）
        - system/user 内容: loss_mask = 0（不计算）
        - assistant 内容: loss_mask = 1（计算 loss）
        - <|im_end|>: loss_mask = 1（计算 loss）
        - <|im_end|> 后的 \\n: loss_mask = 0（不计算）
        
        Returns:
            (token_ids, loss_mask)
        """
        item = self.data[idx]
        
        # 统一转换为 messages 格式
        if "conversations" in item:
            messages = item["conversations"]
        elif "messages" in item:
            messages = item["messages"]
        else:
            messages = [
                {"role": "user", "content": item.get("instruction", "") + item.get("input", "")},
                {"role": "assistant", "content": item.get("output", "")},
            ]
        
        # 确保有 system message
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
        
        # 一次性编码整个对话，得到完整 token ids
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        
        if isinstance(full_ids, dict):
            full_ids = full_ids["input_ids"]
        elif hasattr(full_ids, "input_ids"):
            full_ids = full_ids.input_ids
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        full_ids = [int(x) for x in full_ids]
        
        # 构建 loss_mask：逐段解析 Qwen 的 ChatML 格式
        # 格式: <|im_start|>role\n content <|im_end|>\n
        loss_mask = [0] * len(full_ids)
        
        i = 0
        while i < len(full_ids):
            # 查找 <|im_start|>
            if full_ids[i] == self.im_start_id:
                # <|im_start|> 算 loss
                loss_mask[i] = 1
                i += 1
                
                # 识别角色：匹配 "role\n" 的 token 序列
                role = None
                for r, r_ids in self.role_tokens.items():
                    r_len = len(r_ids)
                    if i + r_len <= len(full_ids) and full_ids[i:i+r_len] == r_ids:
                        role = r
                        break
                
                if role is not None:
                    # 角色名 + \n：不算 loss
                    r_len = len(self.role_tokens[role])
                    for j in range(i, i + r_len):
                        loss_mask[j] = 0
                    i += r_len
                    
                    # 内容部分：到 <|im_end|> 为止
                    while i < len(full_ids) and full_ids[i] != self.im_end_id:
                        if role == "assistant":
                            loss_mask[i] = 1  # assistant 内容算 loss
                        else:
                            loss_mask[i] = 0  # system/user 内容不算
                        i += 1
                    
                    # <|im_end|> 算 loss
                    if i < len(full_ids) and full_ids[i] == self.im_end_id:
                        loss_mask[i] = 1
                        i += 1
                    
                    # <|im_end|> 后的 \n 不算 loss
                    if i < len(full_ids) and self.nl_id and full_ids[i] == self.nl_id[0]:
                        loss_mask[i] = 0
                        i += 1
                else:
                    # 无法识别角色，跳过这个 token
                    i += 1
            else:
                # 其他位置默认不算
                loss_mask[i] = 0
                i += 1
        
        # 确保有 EOS
        eos = self.tokenizer.eos_token_id
        if eos is not None and (len(full_ids) == 0 or full_ids[-1] != eos):
            full_ids.append(eos)
            loss_mask.append(1)  # EOS 算 loss
        
        return full_ids, loss_mask


def sft_data_generator_bos_bestfit(
    dataset: SFTDataset,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    batch_size: int,
    device: str,
    buffer_size: int = 100,
):
    """
    BOS-aligned dataloader with bestfit-pad packing + Qwen label mask.
    
    Yields:
        (inputs, targets): 
            - inputs: (batch_size, max_seq_len) int32
            - targets: (batch_size, max_seq_len) int64
              - 有 loss 的位置: next token id
              - 无 loss 的位置: -100 (ignore_index)
    """
    assert len(dataset) > 0, "Dataset must not be empty"
    
    row_capacity = max_seq_len + 1
    
    bos_token = tokenizer.bos_token_id
    if bos_token is None:
        bos_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    assert bos_token is not None, "Tokenizer must have bos_token, pad_token, or eos_token"
    
    # 缓冲区：(tokens, mask) 元组
    conv_buffer = []
    cursor = 0
    
    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size and cursor < len(dataset):
            tokens, mask = dataset.get_conversation_tokens_with_mask(cursor)
            if len(tokens) > 0:
                conv_buffer.append((tokens, mask))
            cursor += 1
        if cursor >= len(dataset) and len(conv_buffer) < buffer_size:
            cursor = 0
    
    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        
        for _ in range(batch_size):
            row = []
            mask_row = []
            padded = False
            
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                
                if len(conv_buffer) == 0:
                    break
                
                remaining = row_capacity - len(row)
                
                # 最佳适配算法
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    cl = len(conv)
                    if cl <= remaining and cl > best_len:
                        best_idx = i
                        best_len = cl
                
                if best_idx >= 0:
                    conv, mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
        
        # 构建张量
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        
        # 应用 Qwen label mask
        for i in range(len(row_lengths)):
            # mask_rows[i][1:] 对应 targets 的每个位置
            row_mask = mask_rows[i][1:]
            content_len = row_lengths[i]
            
            for j in range(min(len(row_mask), targets.size(1))):
                if row_mask[j] == 0:
                    targets[i, j] = -100
            
            # padding 区域也 mask
            if content_len < row_capacity:
                targets[i, content_len-1:] = -100
        
        yield inputs, targets


# =============================================================================
# 简化版 DataLoader - 无 packing + Qwen label mask
# =============================================================================

def sft_data_generator_simple(
    dataset: SFTDataset,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    batch_size: int,
    device: str,
    shuffle: bool = True,
):
    """
    简化版 SFT dataloader：每行一个对话 + Qwen 官方 label mask
    
    Yields:
        (inputs, targets): 
            - inputs: (batch_size, max_seq_len) int32
            - targets: (batch_size, max_seq_len) int64
    """
    assert len(dataset) > 0, "Dataset must not be empty"
    
    effective_len = max_seq_len + 1
    
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id
    assert pad_token is not None
    
    indices = list(range(len(dataset)))
    
    while True:
        if shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            rows = []
            mask_rows = []
            content_lengths = []
            
            for idx in batch_indices:
                tokens, mask = dataset.get_conversation_tokens_with_mask(idx)
                
                if len(tokens) > effective_len:
                    tokens = tokens[:effective_len]
                    mask = mask[:effective_len]
                
                content_len = len(tokens)
                
                if content_len < effective_len:
                    pad_len = effective_len - content_len
                    tokens = tokens + [pad_token] * pad_len
                    mask = mask + [0] * pad_len
                
                rows.append(tokens)
                mask_rows.append(mask)
                content_lengths.append(content_len)
            
            while len(rows) < batch_size:
                rows.append(rows[-1] if rows else [pad_token] * effective_len)
                mask_rows.append(mask_rows[-1] if mask_rows else [0] * effective_len)
                content_lengths.append(content_lengths[-1] if content_lengths else 0)
            
            batch_tensor = torch.tensor(rows, dtype=torch.long)
            
            inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32)
            targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64)
            
            # 应用 Qwen label mask
            for i in range(len(content_lengths)):
                row_mask = mask_rows[i][1:]
                content_len = content_lengths[i]
                
                for j in range(min(len(row_mask), targets.size(1))):
                    if row_mask[j] == 0:
                        targets[i, j] = -100
                
                if content_len < effective_len:
                    targets[i, content_len-1:] = -100
            
            yield inputs, targets