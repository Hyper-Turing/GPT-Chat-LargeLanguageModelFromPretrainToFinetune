"""
SFT 数据生成器 - BOS 对齐 + 最佳适配打包 + Role Mask

基于 NanoChat/chat_sft.py 中的 sft_data_generator_bos_bestfit 实现，
适配当前项目的 GPT 模型框架，添加 Role-based Loss Mask。

核心功能：
1. BOS 对齐：每个批次行以 BOS 开始
2. 最佳适配打包：选择能完全放入剩余空间的最大对话
3. 填充而非裁剪：使用 BOS token 填充，不丢弃数据
4. 目标掩码：user/system 部分 mask（targets = -100），只计算 assistant loss
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
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_conversation_tokens(self, idx: int) -> list[int]:
        """
        获取指定对话的完整 token IDs（包含 BOS/EOS）
        
        Args:
            idx: 数据索引
            
        Returns:
            token ID 列表
        """
        tokens, _ = self.get_conversation_tokens_with_mask(idx)
        return tokens
    
    def get_conversation_tokens_with_mask(self, idx: int) -> tuple[list[int], list[int]]:
        """
        获取指定对话的 token IDs 和 loss mask
        
        Args:
            idx: 数据索引
            
        Returns:
            (token_ids, loss_mask): 
                - token_ids: token ID 列表
                - loss_mask: 每个位置是否计算 loss，1=计算，0=mask
                  只有 assistant 角色的内容计算 loss，user/system 被 mask
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
        
        # 逐条处理，标记每个 token 的归属
        all_token_ids = []
        all_loss_mask = []  # 1=计算loss, 0=mask
        
        for i, msg in enumerate(messages):
            # 构建到当前消息为止的完整对话
            partial_msgs = messages[:i + 1]
            is_last = (i == len(messages) - 1)
            is_user = (msg["role"] == "user")
            
            # 编码完整对话
            result = self.tokenizer.apply_chat_template(
                partial_msgs,
                tokenize=True,
                add_generation_prompt=(is_last and is_user),
                return_tensors=None,
            )
            
            # 提取 token ids
            if isinstance(result, dict):
                curr_ids = result["input_ids"]
            elif hasattr(result, "input_ids"):
                curr_ids = result.input_ids
            else:
                curr_ids = result
            
            if curr_ids and isinstance(curr_ids[0], list):
                curr_ids = curr_ids[0]
            
            curr_ids = [int(x) for x in curr_ids]
            
            # 计算新增的 tokens
            prev_len = len(all_token_ids)
            new_ids = curr_ids[prev_len:]
            
            # 判断是否应该计算 loss
            # assistant 的内容计算 loss，user/system 被 mask
            # assistant 的 <im_start>assistant\n 标签本身也要 mask，只保留内容
            should_compute_loss = (msg["role"] == "assistant")
            
            if should_compute_loss and len(new_ids) > 0:
                # 找到 assistant 标签的结束位置（\n 之后）
                # Qwen 的格式：<|im_start|>assistant\n内容<|im_end|>
                # 需要把 <|im_start|>assistant\n 这部分 mask 掉
                
                # 简单策略：assistant 消息的第一个换行符之前都 mask
                # 如果没有换行符，则只 mask 第一个 token（<|im_start|>）
                mask_until = 0
                for j, tid in enumerate(new_ids):
                    token_text = self.tokenizer.decode([tid], skip_special_tokens=False)
                    if '\n' in token_text:
                        mask_until = j + 1
                        break
                    mask_until = j + 1  # 至少 mask 第一个 <|im_start|>
                
                for j, _ in enumerate(new_ids):
                    all_token_ids.append(_)
                    if j < mask_until:
                        all_loss_mask.append(0)  # mask 标签部分
                    else:
                        all_loss_mask.append(1)  # 内容部分计算 loss
            else:
                # user/system 全部 mask
                for _ in new_ids:
                    all_token_ids.append(_)
                    all_loss_mask.append(0)
        
        # 确保长度一致
        assert len(all_token_ids) == len(all_loss_mask)
        
        # 添加 EOS（如果还没有的话）
        eos = self.tokenizer.eos_token_id
        if eos is not None and (len(all_token_ids) == 0 or all_token_ids[-1] != eos):
            all_token_ids.append(eos)
            all_loss_mask.append(1 if messages[-1]["role"] == "assistant" else 0)
        
        return all_token_ids, all_loss_mask


def sft_data_generator_bos_bestfit(
    dataset: SFTDataset,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    batch_size: int,
    device: str,
    buffer_size: int = 100,
):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing and role mask.
    
    每个批次行以 BOS 开始，对话使用最佳适配算法打包。
    当没有对话完全适配时，使用 BOS token 填充。
    只计算 assistant 内容的 loss（user/system 被 mask 为 -1）。
    
    Args:
        dataset: SFTDataset 实例
        tokenizer: 分词器
        max_seq_len: 最大序列长度（输入长度）
        batch_size: 批次大小
        device: 设备
        buffer_size: 对话缓冲区大小
    
    Yields:
        (inputs, targets): 
            - inputs: (batch_size, max_seq_len) 输入 token IDs，int32
            - targets: (batch_size, max_seq_len) 目标 token IDs，int64
              - assistant 内容: 实际的 next token id
              - user/system 内容: -100（ignore_index）
              - 填充位置: -100（ignore_index）
    """
    assert len(dataset) > 0, "Dataset must not be empty"
    
    # row_capacity = max_seq_len + 1 因为需要 targets 有 max_seq_len 个元素
    # inputs = row[:-1], targets = row[1:]
    row_capacity = max_seq_len + 1
    
    # 使用 BOS token 作为填充
    bos_token = tokenizer.bos_token_id
    if bos_token is None:
        bos_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    assert bos_token is not None, "Tokenizer must have bos_token, pad_token, or eos_token"
    
    # 对话缓冲区：(tokens, mask) 元组列表
    conv_buffer = []
    cursor = 0
    
    def refill_buffer():
        """从数据集中填充缓冲区"""
        nonlocal cursor
        while len(conv_buffer) < buffer_size and cursor < len(dataset):
            conv_tokens, conv_mask = dataset.get_conversation_tokens_with_mask(cursor)
            if len(conv_tokens) > 0:
                conv_buffer.append((conv_tokens, conv_mask))
            cursor += 1
        
        # 数据集用完后重置游标（无限循环）
        if cursor >= len(dataset) and len(conv_buffer) < buffer_size:
            cursor = 0
    
    while True:
        rows = []           # token IDs
        mask_rows = []      # loss mask (1=计算, 0=mask)
        row_lengths = []    # 实际内容长度
        
        for _ in range(batch_size):
            row = []
            mask_row = []
            padded = False
            
            while len(row) < row_capacity:
                # 确保缓冲区有足够对话
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                
                if len(conv_buffer) == 0:
                    break
                
                remaining = row_capacity - len(row)
                
                # 【最佳适配算法】查找能完全适配的最大对话
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                
                if best_idx >= 0:
                    # 使用适配的对话
                    conv, mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(mask)
                else:
                    # 【填充策略】没有对话适配，使用 BOS 填充
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)  # 填充部分 mask
                    padded = True
                    break
            
            # 记录内容长度
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
        
        # 构建张量
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        
        # inputs: (batch, max_seq_len), targets: (batch, max_seq_len)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        
        # 应用 role mask 和填充 mask
        for i, content_len in enumerate(row_lengths):
            # 获取当前行的 mask（去掉第一个位置，因为 targets 是右移的）
            row_mask = mask_rows[i][1:]  # 去掉第一个，对应 targets 的位置
            
            # 将 mask=0 的位置设为 -100（ignore_index）
            for j, m in enumerate(row_mask):
                if j < targets.size(1) and m == 0:
                    targets[i, j] = -100
            
            # 填充位置也设为 -100（从 content_len-1 开始）
            if content_len < row_capacity:
                targets[i, content_len-1:] = -100
        
        yield inputs, targets


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 测试配置
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    DATA_PATH = "data/sft/sft_train.jsonl"
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 2
    
    print("=" * 70)
    print("SFT 数据生成器测试 (带 Role Mask)")
    print("=" * 70)
    
    if not os.path.exists(DATA_PATH):
        print(f"数据文件不存在: {DATA_PATH}")
        exit(1)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    dataset = SFTDataset(DATA_PATH, tokenizer)
    print(f"数据集: {len(dataset)} 条对话")
    
    # 检查 tokenizer
    print(f"\nToken IDs: bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")
    
    # 查看第一条对话的 role mask
    print("\n" + "=" * 70)
    print("第一条对话的 Role Mask 示例:")
    print("=" * 70)
    tokens, mask = dataset.get_conversation_tokens_with_mask(0)
    print(f"总长度: {len(tokens)} tokens")
    print(f"计算 loss 的位置数: {sum(mask)}")
    print(f"被 mask 的位置数: {len(mask) - sum(mask)}")
    
    # 解码查看
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    print(f"\n解码文本:\n{text[:500]}{'...' if len(text) > 500 else ''}")
    
    # 显示哪些部分计算 loss
    print(f"\n角色标记 (1=assistant计算loss, 0=user/system被mask):")
    # 分段显示
    for i in range(0, min(len(mask), 50), 10):
        seg_tokens = tokens[i:i+10]
        seg_mask = mask[i:i+10]
        seg_text = tokenizer.decode(seg_tokens, skip_special_tokens=False)
        print(f"  [{i:3d}:{i+10:3d}] mask={seg_mask} text={repr(seg_text[:30])}")
    
    # 创建数据生成器
    device = "cpu"
    loader = sft_data_generator_bos_bestfit(
        dataset, tokenizer, MAX_SEQ_LEN, BATCH_SIZE, device, buffer_size=10
    )
    
    # 获取一个 batch
    inputs, targets = next(loader)
    
    print(f"\n{'=' * 70}")
    print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}")
    print(f"{'=' * 70}")
    
    # 处理第0条数据
    input_ids = inputs[0].cpu().tolist()
    target_ids = targets[0].cpu().tolist()
    
    # 统计（整个序列）
    n_ignore = sum(1 for t in target_ids if t == -100)
    n_compute = len(target_ids) - n_ignore
    
    print(f"\n第0条数据统计:")
    print(f"  总长度: {len(target_ids)} tokens")
    print(f"  计算 loss: {n_compute} tokens")
    print(f"  被 mask (-100): {n_ignore} tokens")
    
    # 找到实际内容结束位置（最后一个非-100的位置）
    content_end = len(target_ids)
    for i in range(len(target_ids) - 1, -1, -1):
        if target_ids[i] != -100:
            content_end = i + 1
            break
    
    # 解码 inputs（完整）
    print(f"\n{'=' * 70}")
    print("INPUTS (第0条):")
    print(f"{'=' * 70}")
    print(tokenizer.decode(input_ids[:content_end], skip_special_tokens=False))
    
    # 解码 targets，只显示计算 loss 的部分（跳过 -100）
    print(f"\n{'=' * 70}")
    print("TARGETS (第0条, 只显示计算loss的部分):")
    print(f"{'=' * 70}")
    target_with_mask = []
    for i, (tid, iid) in enumerate(zip(target_ids[:content_end], input_ids[1:content_end+1])):
        if tid == -100:
            target_with_mask.append(f"[mask]")
        else:
            # 解码单个 token
            token_text = tokenizer.decode([iid], skip_special_tokens=False)
            target_with_mask.append(token_text)
    
    # 合并显示
    result = ""
    for t in target_with_mask:
        if t == "[mask]":
            result += "▯"  # 用特殊符号表示 mask
        else:
            result += t
    print(result if result else "(无)")
    
    # 单独显示计算 loss 的 tokens
    loss_tokens = [input_ids[i+1] for i, tid in enumerate(target_ids[:content_end]) if tid != -100]
    if loss_tokens:
        print(f"\n实际计算loss的文本:")
        print(tokenizer.decode(loss_tokens, skip_special_tokens=False))
    else:
        print(f"\n没有计算loss的token")
