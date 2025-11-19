#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute statistics for HW4 Q4 using T5-small tokenizer.

- Table 1: Before preprocessing
- Table 2: After preprocessing (using the same preprocessing as your T5 training)

Usage:
    python q4_stats.py
"""

import os
from typing import List, Tuple, Dict

from transformers import T5TokenizerFast
from load_data import preprocess_input, preprocess_output


# ===================== Config =====================

DATA_DIR = "data"

TRAIN_NL_PATH = os.path.join(DATA_DIR, "train.nl")
TRAIN_SQL_PATH = os.path.join(DATA_DIR, "train.sql")
DEV_NL_PATH = os.path.join(DATA_DIR, "dev.nl")
DEV_SQL_PATH = os.path.join(DATA_DIR, "dev.sql")

T5_MODEL_NAME = "google-t5/t5-small"


# ===================== Utility functions =====================

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_length_and_vocab(
    lines: List[str],
    tokenizer: T5TokenizerFast
) -> Dict[str, float]:
    """
    对一组文本：
        - 计算每个样本的 token 长度（T5 vocab 下）
        - 统计所有样本中出现过的 token id 的种类数（vocab size）
    """
    lengths = []
    vocab = set()

    for line in lines:
        if line == "":
            # 空行也可以直接跳过
            continue
        token_ids = tokenizer(line).input_ids
        lengths.append(len(token_ids))
        vocab.update(token_ids)

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "avg_len": avg_len,
        "vocab_size": len(vocab)
    }


def print_table_1(
    train_nl_stats,
    train_sql_stats,
    dev_nl_stats,
    dev_sql_stats,
    num_train,
    num_dev
):
    print("=" * 60)
    print("Table 1: Dataset statistics BEFORE preprocessing")
    print("=" * 60)
    print(f"Number of training examples: {num_train}")
    print(f"Number of dev examples:      {num_dev}")
    print()
    print("Average length (in T5 tokens):")
    print(f"  Train NL:  {train_nl_stats['avg_len']:.2f}")
    print(f"  Train SQL: {train_sql_stats['avg_len']:.2f}")
    print(f"  Dev NL:    {dev_nl_stats['avg_len']:.2f}")
    print(f"  Dev SQL:   {dev_sql_stats['avg_len']:.2f}")
    print()
    print("Vocabulary size (number of unique token IDs):")
    print(f"  Train NL:  {train_nl_stats['vocab_size']}")
    print(f"  Train SQL: {train_sql_stats['vocab_size']}")
    print(f"  Dev NL:    {dev_nl_stats['vocab_size']}")
    print(f"  Dev SQL:   {dev_sql_stats['vocab_size']}")
    print("=" * 60)
    print()


def print_table_2(
    train_nl_stats_pp,
    train_sql_stats_pp,
    dev_nl_stats_pp,
    dev_sql_stats_pp
):
    print("=" * 60)
    print("Table 2: Dataset statistics AFTER preprocessing")
    print("=" * 60)
    print(f"Model name: {T5_MODEL_NAME}")
    print()
    print("Average length (in T5 tokens) after preprocessing:")
    print(f"  Train NL (preprocessed):  {train_nl_stats_pp['avg_len']:.2f}")
    print(f"  Train SQL (preprocessed): {train_sql_stats_pp['avg_len']:.2f}")
    print(f"  Dev NL (preprocessed):    {dev_nl_stats_pp['avg_len']:.2f}")
    print(f"  Dev SQL (preprocessed):   {dev_sql_stats_pp['avg_len']:.2f}")
    print()
    print("Vocabulary size after preprocessing:")
    print(f"  Train NL (preprocessed):  {train_nl_stats_pp['vocab_size']}")
    print(f"  Train SQL (preprocessed): {train_sql_stats_pp['vocab_size']}")
    print(f"  Dev NL (preprocessed):    {dev_nl_stats_pp['vocab_size']}")
    print(f"  Dev SQL (preprocessed):   {dev_sql_stats_pp['vocab_size']}")
    print("=" * 60)
    print()


# ===================== Main =====================

def main():
    # 1. 加载 tokenizer
    print(f"Loading tokenizer: {T5_MODEL_NAME}")
    tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_NAME)

    # 2. 读取原始数据
    print("Reading data files...")
    train_nl = read_lines(TRAIN_NL_PATH)
    train_sql = read_lines(TRAIN_SQL_PATH)
    dev_nl = read_lines(DEV_NL_PATH)
    dev_sql = read_lines(DEV_SQL_PATH)

    assert len(train_nl) == len(train_sql), "train.nl 和 train.sql 行数不一致！"
    assert len(dev_nl) == len(dev_sql), "dev.nl 和 dev.sql 行数不一致！"

    num_train = len(train_nl)
    num_dev = len(dev_nl)

    # 3. BEFORE preprocessing stats (Table 1)
    print("Computing statistics BEFORE preprocessing...")
    train_nl_stats = get_length_and_vocab(train_nl, tokenizer)
    train_sql_stats = get_length_and_vocab(train_sql, tokenizer)
    dev_nl_stats = get_length_and_vocab(dev_nl, tokenizer)
    dev_sql_stats = get_length_and_vocab(dev_sql, tokenizer)

    print_table_1(
        train_nl_stats,
        train_sql_stats,
        dev_nl_stats,
        dev_sql_stats,
        num_train,
        num_dev
    )

    # 4. AFTER preprocessing stats (Table 2)
    print("Applying preprocessing defined in preprocess_input / preprocess_output...")
    train_nl_pp = [preprocess_input(x) for x in train_nl]
    dev_nl_pp = [preprocess_input(x) for x in dev_nl]
    train_sql_pp = [preprocess_output(x) for x in train_sql]
    dev_sql_pp = [preprocess_output(x) for x in dev_sql]

    print("Computing statistics AFTER preprocessing...")
    train_nl_stats_pp = get_length_and_vocab(train_nl_pp, tokenizer)
    train_sql_stats_pp = get_length_and_vocab(train_sql_pp, tokenizer)
    dev_nl_stats_pp = get_length_and_vocab(dev_nl_pp, tokenizer)
    dev_sql_stats_pp = get_length_and_vocab(dev_sql_pp, tokenizer)

    print_table_2(
        train_nl_stats_pp,
        train_sql_stats_pp,
        dev_nl_stats_pp,
        dev_sql_stats_pp
    )

    print("Done. Copy the numbers above into Table 1 and Table 2 in your report.")


if __name__ == "__main__":
    main()
