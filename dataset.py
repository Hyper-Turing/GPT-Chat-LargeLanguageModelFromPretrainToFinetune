"""
dataset.py — SFT 数据集（重写版）

核心改动：
  1. 使用 tokenizer.apply_chat_template 做 tokenization，
     与推理时 test_llm.py 使用完全相同的方法，确保训练/推理一致
  2. 通过增量 tokenization 精确定位 assistant 边界，构建 loss mask
  3. padding 使用 eos_token_id (151643)，ChatML 格式中不会出现此 token

设计原则：
  - 不手动拼接 <|im_start|>/<|im_end|>，全部交给 chat template
  - loss mask 只标记 assistant 的 content + <|im_end|>
  - 简单直接，无 packing，一条对话一个样本
"""

import json
from typing import List, Tuple, Optional

import torch

IGNORE_TOKEN_ID = -100


class SFTDataset:
    """
    SFT 对话数据集

    支持格式:
      - {"conversations": [{"role": "...", "content": "..."}, ...]}
      - {"messages": [{"role": "...", "content": "..."}, ...]}
      - {"instruction": "...", "input": "...", "output": "..."}
    """

    def __init__(self, filepath: str, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer

        # Qwen ChatML 特殊 token IDs
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = tokenizer.eos_token_id  # 151643 <|endoftext|>

        # 验证 special tokens 存在
        assert self.im_start_id != tokenizer.unk_token_id, \
            "Tokenizer 缺少 <|im_start|> token，请确认使用 Qwen 系列 tokenizer"
        assert self.im_end_id != tokenizer.unk_token_id, \
            "Tokenizer 缺少 <|im_end|> token，请确认使用 Qwen 系列 tokenizer"

        # 加载数据
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def _to_id_list(self, result) -> list:
        """确保 apply_chat_template 的返回值是 list[int]
        
        不同版本的 transformers / tokenizer 可能返回:
        - list[int]       (正常)
        - BatchEncoding   (某些版本)
        - torch.Tensor    (指定 return_tensors 时)
        """
        if isinstance(result, list):
            return result
        if hasattr(result, 'input_ids'):          # BatchEncoding / dict
            ids = result['input_ids']
            return ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        if hasattr(result, 'tolist'):             # torch.Tensor / np.array
            return result.tolist()
        return list(result)

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_messages(self, item) -> list:
        """统一不同格式的对话数据为 messages 列表"""
        if "conversations" in item:
            messages = item["conversations"]
        elif "messages" in item:
            messages = item["messages"]
        else:
            user_text = (item.get("instruction", "") or "")
            if item.get("input"):
                user_text += "\n" + item["input"]
            messages = [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": item.get("output", "") or ""},
            ]

        # 确保有 system 消息
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages

        return messages

    def get_conversation_tokens(self, idx: int) -> List[int]:
        """获取对话的 token 序列（不含 mask）"""
        tokens, _ = self.get_conversation_tokens_with_mask(idx)
        return tokens

    def get_conversation_tokens_with_mask(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        获取对话的 token 序列和 loss mask

        使用 apply_chat_template 做 tokenization，保证与推理时完全一致。
        通过增量 tokenization 精确定位每个 assistant 回复的边界。

        Returns:
            token_ids: 完整对话的 token 列表
            loss_mask: 与 token_ids 等长，1=计算 loss，0=忽略
                       只有 assistant 的 content + <|im_end|> 为 1
        """
        item = self.data[idx]
        messages = self._normalize_messages(item)

        # ================================================================
        # Step 1: 用 apply_chat_template 获取完整 token 序列
        # ================================================================
        full_ids = self._to_id_list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False
        ))

        # ================================================================
        # Step 2: 增量 tokenization 定位 assistant 边界
        # ================================================================
        loss_mask = [0] * len(full_ids)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # 获取此 assistant 回复之前的 token 数（含 generation prompt）
            prefix_msgs = messages[:i]
            prefix_ids = self._to_id_list(self.tokenizer.apply_chat_template(
                prefix_msgs, tokenize=True, add_generation_prompt=True,
                add_special_tokens=False,
            ))
            prefix_len = len(prefix_ids)

            # 获取包含此 assistant 回复的 token 数
            through_msgs = messages[:i + 1]
            through_ids = self._to_id_list(self.tokenizer.apply_chat_template(
                through_msgs, tokenize=True, add_generation_prompt=False,
                add_special_tokens=False,
            ))
            through_len = len(through_ids)

            # assistant 的 tokens 范围: [prefix_len, through_len)
            # 这包含了 content + <|im_end|> + 可能的 \n
            #
            # 我们要标记 content + <|im_end|> 为 mask=1
            # 不标记末尾的 \n（如果有的话）
            end_pos = through_len

            # 检查末尾是否有 \n，如果有则不训练它
            nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if end_pos >= len(nl_ids):
                trailing = full_ids[end_pos - len(nl_ids):end_pos]
                if trailing == nl_ids:
                    end_pos -= len(nl_ids)

            # 标记 assistant content + <|im_end|>
            for j in range(prefix_len, min(end_pos, len(full_ids))):
                loss_mask[j] = 1

        # ================================================================
        # Step 3: 处理序列末尾
        # ================================================================
        # 如果最后一条消息是 assistant，确保序列以 <|im_end|> 结尾
        # 移除 apply_chat_template 可能追加的末尾 \n
        if messages[-1]["role"] == "assistant":
            nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            while (len(full_ids) >= len(nl_ids) and
                   full_ids[-len(nl_ids):] == nl_ids and
                   full_ids[-len(nl_ids) - 1] == self.im_end_id):
                # 移除 <|im_end|> 后面多余的 \n
                full_ids = full_ids[:-len(nl_ids)]
                loss_mask = loss_mask[:-len(nl_ids)]

        return full_ids, loss_mask


def sft_data_generator(
    dataset: SFTDataset,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    device: str,
):
    """
    SFT 数据生成器（简单版，不 packing）

    Yields:
        (inputs, targets, attn_mask):
            inputs:    (B, max_seq_len) int32 — 模型输入
            targets:   (B, max_seq_len) int64 — 训练目标（-100 表示忽略）
            attn_mask: (B, max_seq_len) bool  — True 表示 padding 位置

    关键设计：
      - PAD_ID = eos_token_id (151643)，ChatML 中不会出现此 token
      - inputs = tokens[:-1], targets = tokens[1:]（标准 next-token prediction）
      - loss mask 对齐到 targets 位置
    """
    assert len(dataset) > 0, "Dataset is empty"

    # Padding token: 使用 eos_token_id (151643 = <|endoftext|>)
    # 在 ChatML 格式中，对话只使用 <|im_start|> 和 <|im_end|>
    # <|endoftext|> 不会出现在正常对话中，用它做 padding 是安全的
    PAD_ID = tokenizer.eos_token_id
    assert PAD_ID is not None, "tokenizer.eos_token_id is None"

    cursor = 0
    seq_plus_one = max_seq_len + 1

    while True:
        batch_tokens = []
        batch_masks = []

        for _ in range(batch_size):
            tokens, mask = dataset.get_conversation_tokens_with_mask(cursor)
            batch_tokens.append(tokens)
            batch_masks.append(mask)
            cursor = (cursor + 1) % len(dataset)

        rows = []
        mask_rows = []

        for tokens, mask in zip(batch_tokens, batch_masks):
            # 截断或填充到 max_seq_len + 1
            if len(tokens) >= seq_plus_one:
                row = tokens[:seq_plus_one]
                row_mask = mask[:seq_plus_one]
            else:
                pad_len = seq_plus_one - len(tokens)
                row = tokens + [PAD_ID] * pad_len
                row_mask = mask + [0] * pad_len

            rows.append(row)
            mask_rows.append(row_mask)

        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)

        # inputs = tokens[0..T-1], targets = tokens[1..T]
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Attention mask: True = padding position (被 mask 的位置)
        attn_mask = inputs.eq(PAD_ID)

        # 应用 loss mask 到 targets
        # mask_rows[i][j] 对应 token j 的 mask
        # targets[i][j] 对应预测 token j+1
        # 所以 mask_rows[i][j+1] 决定 targets[i][j] 是否参与 loss
        for i in range(batch_size):
            row_mask = mask_rows[i]
            for j in range(targets.size(1)):
                # targets[i][j] 是预测 row[j+1] 的目标
                # 如果 row[j+1] 不需要计算 loss，则设为 -100
                if j + 1 < len(row_mask) and row_mask[j + 1] == 0:
                    targets[i, j] = IGNORE_TOKEN_ID

        yield inputs, targets, attn_mask


# 兼容旧接口名
sft_data_generator_single = sft_data_generator