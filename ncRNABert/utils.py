""" 
"""
from typing import Sequence, Tuple, List, Union
from torch.nn.utils.rnn import pad_sequence
import torch
import pathlib
import urllib
import re


base_rna_tokens = list("GAUCRYWSMKN") 
special_rna_tokens = ['<UNK>', 'TOKEN_MASK', 'PADDING_MASK']

rna_tokens = base_rna_tokens + special_rna_tokens


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


def BatchConverter(raw_batch: Sequence[Tuple[str, str]]):
    ids = [item[0] for item in raw_batch]
    seqs = [item[1] for item in raw_batch]
    batch_token = []
    for seq in seqs:
        seq = re.sub(r"[T]", "U", seq.upper())
        batch_token.append(torch.tensor([token_to_index.get(i, token_to_index["<UNK>"]) for i in seq]))
    batch_token = pad_sequence(batch_token, batch_first=True, padding_value=token_to_index['PADDING_MASK'])
    return ids, batch_token


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