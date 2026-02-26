"""
SFT 数据集
数据格式: {"conversations": [{"role": "...", "content": "..."}, ...]}
ChatML format
<im_start>systemYou are a helpful assistant.<im_end>
<im_start>user
Hello!<|im_end>
<im_start>assistant
Hello!How can I assist you today?<im_end>

使用 bestfit packing 算法提高batch内token利用率
"""
import json
import random
from typing import List, Tuple

import torch

IGNORE_TOKEN_ID = -100


class SFTDataset:
    def __init__(self, filepath: str, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        # 使用字符串字面量替代特殊字符
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.data)

    def get_tokens_with_mask(self, idx: int):
        """
        获取对话的 token 序列和 label mask
        label_mask: 
            1=计算loss: assistant content + <|im_end|>
            0=忽略
        """
        item = self.data[idx]
        messages = item["conversations"]

        # 返回 BatchEncoding(字典), 需要提取 input_ids
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False
        )["input_ids"]

        # 定位 assistant 边界
        loss_mask = [0] * len(full_ids)
        # 换行符id 用于后续去末尾\n
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # assistant回复之前的消息
            prefix_ids = self.tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True,
                add_special_tokens=False,
            )["input_ids"]
            prefix_len = len(prefix_ids) # assistant回复开始的位置

            # assistant回复的消息
            assistant_ids = self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=True, add_generation_prompt=False,
                add_special_tokens=False,
            )["input_ids"]
            end_pos = len(assistant_ids) # assistant回复结束的位置

            # chatML 中, assistant的回复是:
            # <|im_start|>assistant
            # 回复内容<|im_end|>\n
            # \n不是回复的一部分, 不需要计算loss
            if end_pos >= len(nl_ids):
                if full_ids[end_pos - len(nl_ids):end_pos] == nl_ids:
                    end_pos -= len(nl_ids)

            # 标记 assistant content + <|im_end|>
            for j in range(prefix_len, min(end_pos, len(full_ids))):
                loss_mask[j] = 1

        # apply_chat_template 可能追加的末尾 \n 也需要删除 
        if messages[-1]["role"] == "assistant":
            while (len(full_ids) >= len(nl_ids) and
                   full_ids[-len(nl_ids):] == nl_ids and
                   full_ids[-len(nl_ids) - 1] == self.im_end_id):
                full_ids = full_ids[:-len(nl_ids)]
                loss_mask = loss_mask[:-len(nl_ids)]

        return full_ids, loss_mask

# bestfit packing
def sft_data_generator(
    dataset: SFTDataset,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    device: str,
    buffer_size: int = 128,
    shuffle: bool = True,
):
    """
    维护一个预tokenize的对话缓冲区，构建每个序列时用besfit算法
    从缓冲区选择能完全放入剩余空间的最大对话 打包

    Returns:
        (inputs, targets, attn_mask)
    """
    assert len(dataset) > 0, "Dataset is empty"

    PAD_ID = tokenizer.eos_token_id
    assert PAD_ID is not None
    max_seq_len = max_seq_len + 1  # +1是因为 inputs=tokens[:-1], targets=tokens[1:]

    # 缓冲区存储 (tokens, mask, length) 三元组
    conv_buffer: list = []
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    cursor = 0
    epoch = 0

    def _next_index():
        # 提前预取下一个数据索引
        # 自动循环防止超出数据集
        nonlocal cursor, epoch
        if cursor >= len(indices):
            cursor = 0
            epoch += 1
            if shuffle:
                random.shuffle(indices)
        idx = indices[cursor]
        cursor += 1
        return idx

    def _refill_buffer():
        # 填充buffer
        while len(conv_buffer) < buffer_size:
            idx = _next_index()
            tokens, mask = dataset.get_tokens_with_mask(idx)
            if not tokens:
                continue
            # 超长对话截断
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                mask = mask[:max_seq_len]
            conv_buffer.append((tokens, mask, len(tokens)))

    while True:
        batch_rows = []
        batch_mask_rows = []

        for _ in range(batch_size):
            row_tokens = []
            row_mask = []
            remaining = max_seq_len

            while remaining > 0:
                _refill_buffer()
                if not conv_buffer:
                    break

                # 找能完全放入且最大的对话
                best_idx = -1
                best_len = 0
                for i, (_, _, clen) in enumerate(conv_buffer):
                    if clen <= remaining and clen > best_len:
                        best_idx = i
                        best_len = clen

                if best_idx >= 0:
                    tokens, mask, _ = conv_buffer.pop(best_idx)
                    row_tokens.extend(tokens)
                    row_mask.extend(mask)
                    remaining -= len(tokens)
                else:
                    # 没有对话能放入 pad 剩余空间
                    row_tokens.extend([PAD_ID] * remaining)
                    row_mask.extend([0] * remaining)
                    remaining = 0

            # 确保长度精确为 max_seq_len
            if len(row_tokens) < max_seq_len:
                pad_len = max_seq_len - len(row_tokens)
                row_tokens.extend([PAD_ID] * pad_len)
                row_mask.extend([0] * pad_len)

            batch_rows.append(row_tokens[:max_seq_len])
            batch_mask_rows.append(row_mask[:max_seq_len])

        # 构建 tensor
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(batch_rows, dtype=torch.long, pin_memory=use_cuda)

        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Attention mask: True = padding (被 mask 的位置)
        attn_mask = inputs.eq(PAD_ID)

        # 应用 loss mask: mask_rows[j+1] 决定 targets[j] 是否参与 loss
        for i in range(batch_size):
            rm = batch_mask_rows[i]
            for j in range(targets.size(1)):
                if j + 1 < len(rm) and rm[j + 1] == 0:
                    targets[i, j] = IGNORE_TOKEN_ID

        yield inputs, targets, attn_mask