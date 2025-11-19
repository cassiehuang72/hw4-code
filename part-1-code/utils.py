import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

# QWERTY 键盘相邻字母映射，用来模拟真实 typo
KEYBOARD_NEIGHBORS = {
    'q': 'w',
    'w': 'qe',
    'e': 'wr',
    'r': 'et',
    't': 'ry',
    'y': 'tu',
    'u': 'yi',
    'i': 'uo',
    'o': 'ip',
    'p': 'o',
    'a': 's',
    's': 'adw',
    'd': 'sfe',
    'f': 'dgr',
    'g': 'fht',
    'h': 'gjy',
    'j': 'hku',
    'k': 'jli',
    'l': 'k',
    'z': 'x',
    'x': 'zc',
    'c': 'xv',
    'v': 'cb',
    'b': 'vn',
    'n': 'bm',
    'm': 'n',
}

_detokenizer = TreebankWordDetokenizer()


def _apply_typo(word: str) -> str:
    # 只对长度 >= 4 且全字母的词做 typo
    if len(word) < 4 or not word.isalpha():
        return word

    chars = list(word)
    # 选择一个中间位置，避免破坏首尾
    idx = random.randint(1, len(chars) - 2)
    c = chars[idx]
    lower_c = c.lower()

    # 50% 概率做邻键替换，50% 概率做相邻交换
    if random.random() < 0.5:
        # 邻键替换
        if lower_c in KEYBOARD_NEIGHBORS:
            cand = random.choice(KEYBOARD_NEIGHBORS[lower_c])
        else:
            cand = random.choice("abcdefghijklmnopqrstuvwxyz")
        # 保持这个字母本身的大小写
        chars[idx] = cand.upper() if c.isupper() else cand
    else:
        # 相邻交换
        j = idx
        if j == len(chars) - 1:
            j -= 1
        chars[j], chars[j + 1] = chars[j + 1], chars[j]

    new_word = "".join(chars)

    # 尽量保持整体大小写模式
    if word.isupper():
        new_word = new_word.upper()
    elif word[0].isupper():
        new_word = new_word.capitalize()

    return new_word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # 超参数：每个单词被加 typo 的概率，你可以按需要微调
    WORD_NOISE_PROB = 0.4

    text = example["text"]
    tokens = word_tokenize(text)
    new_tokens = []

    for tok in tokens:
        # 只对“像单词”的 token 做噪声：全字母且长度>=4
        if tok.isalpha() and len(tok) >= 4 and random.random() < WORD_NOISE_PROB:
            new_tok = _apply_typo(tok)
            new_tokens.append(new_tok)
        else:
            new_tokens.append(tok)

    # detokenize 回字符串
    example["text"] = _detokenizer.detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example


