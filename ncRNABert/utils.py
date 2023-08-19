""" 
"""
from typing import Sequence, Tuple, List, Union
from torch.nn.utils.rnn import pad_sequence
import torch
import pathlib
import urllib
import re


nucleotides = "GAUC"
rna_3 = [i + j + k for i in nucleotides for j in nucleotides for k in nucleotides]
rna_2 = [i + j for i in nucleotides for j in nucleotides]
rna_1 = [i for i in nucleotides]

special_rna_tokens = ['<UNK>', 'TOKEN_MASK', 'PADDING_MASK']

rna_tokens = rna_3 + rna_2 + rna_1 + special_rna_tokens


token_to_index = {}
for i, j in enumerate(rna_tokens):
    token_to_index[j] = i

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def split_rna_seq(sequence):
    sequence_length = len(sequence)
    token_split_list = [sequence[i:i+3] for i in range(sequence_length-2)]
    token_split_list += [sequence[-2:], sequence[-1]]
    return token_split_list

def BatchConverter(raw_batch: Sequence[Tuple[str, str]]):
    ids = [item[0] for item in raw_batch]
    seqs = [item[1] for item in raw_batch]
    lengths = torch.tensor([len(item[1]) for item in raw_batch])
    batch_token = []
    for seq in seqs:
        seq = re.sub(r"[T]", "U", seq.upper())
        batch_token.append(torch.tensor([token_to_index.get(i, token_to_index["<UNK>"]) for i in split_rna_seq(seq)]))
    batch_token = pad_sequence(batch_token, batch_first=True, padding_value=token_to_index['PADDING_MASK'])
    return ids, batch_token, lengths


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = pathlib.Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check your network!")
    return 