## ncRNABert: Deciphering the landscape of non-coding RNA using language model

### Model details
|   **Model**    | **# of parameters** | **# of hidden size** |            **Pretraining dataset**             | **# of ncRNAs** | **Model download** |
|:--------------:|:-------------------:|:----------------------:|:----------------------------------------------:|:-----------------:|:------------------------:|
|    ncRNABert   |        303M         |           1024           | [RNAcentral](http://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz) |       26M        |      [Download](https://zenodo.org/)       |

### Install
As a prerequisite, you must have PyTorch installed to use this repository.

You can use this one-liner for installation, using the latest release version

```
# latest version
pip install git+https://github.com/wangleiofficial/ncRNABert

# stable version
pip install ncRNABert

```

### Usage

#### ncRNA sequence embedding

```
from ncRNABert.pretrain import load_ncRNABert
from ncRNABert.utils import BatchConverter
import torch

data = [
    ("ncRNA1", "ACGGAGGATGCGAGCGTTATCCGGATTTACTGGGCG"),
    ("ncRNA2", "AGGTTTTTAATCTAATTAAGATAGTTGA"),
]

ids, batch_token, lengths = BatchConverter(data)
model = load_ncRNANert()
with torch.no_grad():
    results = model(batch_token, lengths, repr_layers=[24])
# Generate per-sequence representations via averaging
token_representations = results["representations"][24]
sequence_representations = []
batch_lens = [len(item[1]) for item in data]
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
```

### License
This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of this source tree.