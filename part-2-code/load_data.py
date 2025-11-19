import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def preprocess_input(nl: str) -> str:
    return "Convert to SQL: " + nl.strip()

def preprocess_output(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = " ".join(sql.split())
    return sql

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        assert split in ["train", "dev", "test"]
        self.split = split
        self.data_folder = data_folder

        # 使用 T5-small tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_id = self.tokenizer.pad_token_id  # 通常是 0

        # 预处理 + tokenization
        self.samples = self.process_data(data_folder, split, self.tokenizer)

        # 用一个 extra-id 作为 decoder 的起始 token（BOS）
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def process_data(self, data_folder, split, tokenizer):
        samples = []

        if split in ["train", "dev"]:
            nl_path = os.path.join(data_folder, f"{split}.nl")
            sql_path = os.path.join(data_folder, f"{split}.sql")

            nl_lines = load_lines(nl_path)
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), f"{split}.nl and {split}.sql have different lengths!"

            for nl, sql in zip(nl_lines, sql_lines):
                # 文本级 preprocessing
                src_text = preprocess_input(nl)
                tgt_text = preprocess_output(sql)

                # 使用同一个 tokenizer 对 encoder / decoder 文本编码
                enc_ids = tokenizer.encode(
                    src_text,
                    add_special_tokens=True,   # T5 会加 EOS
                    return_tensors=None,
                )
                dec_ids = tokenizer.encode(
                    tgt_text,
                    add_special_tokens=True,
                    return_tensors=None,
                )

                # 转成 LongTensor，方便后续 pad_sequence
                enc_ids = torch.tensor(enc_ids, dtype=torch.long)
                dec_ids = torch.tensor(dec_ids, dtype=torch.long)

                samples.append((enc_ids, dec_ids))

        else:  # split == "test"
            nl_path = os.path.join(data_folder, "test.nl")
            nl_lines = load_lines(nl_path)

            for nl in nl_lines:
                src_text = preprocess_input(nl)
                enc_ids = tokenizer.encode(
                    src_text,
                    add_special_tokens=True,
                    return_tensors=None,
                )
                enc_ids = torch.tensor(enc_ids, dtype=torch.long)
                samples.append(enc_ids)

        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # 拆包：batch 是 List[(enc_ids, dec_ids)]
    encoder_seqs, decoder_seqs = zip(*batch)  # 各是长度 B 的 tuple

    # 动态 padding encoder
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)  # B x T
    encoder_mask = (encoder_ids != PAD_IDX).long()  # B x T

    # 动态 padding decoder targets
    decoder_targets = pad_sequence(decoder_seqs, batch_first=True, padding_value=PAD_IDX)  # B x T'

    # 构造 decoder_inputs：在每个序列开头加 BOS，整体右移一位
    # 先拿到 BOS id（从第一条样本的 tokenizer 来）
    # 这里假设 __getitem__ 里保存了 t5 tokenizer 的 bos_id 到对象里，
    # 但 collate_fn 拿不到 dataset，所以我们用一个简单 trick：
    from transformers import T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_id = tok.convert_tokens_to_ids("<extra_id_0>")

    decoder_input_seqs = []
    for seq in decoder_seqs:
        # seq: 1D tensor, 长度 L
        L = seq.size(0)
        # 构造长度 L 的 input: [BOS, y_0, ..., y_{L-2}]
        inp = torch.empty_like(seq)
        inp[0] = bos_id
        if L > 1:
            inp[1:] = seq[:-1]
        decoder_input_seqs.append(inp)

    decoder_inputs = pad_sequence(decoder_input_seqs, batch_first=True, padding_value=PAD_IDX)  # B x T'

    # initial_decoder_inputs: 每个样本的第一个 decoder token（BOS）
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size,), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seqs = batch

    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)  # B x T
    encoder_mask = (encoder_ids != PAD_IDX).long()  # B x T

    from transformers import T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_id = tok.convert_tokens_to_ids("<extra_id_0>")

    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size,), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
