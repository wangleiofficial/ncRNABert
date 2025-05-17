## ncRNABert: Deciphering the landscape of non-coding RNA using language model

[![PyPI - Version](https://img.shields.io/pypi/v/ncRNABert.svg?style=flat)](https://pypi.org/project/ncRNABert/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ncRNABert.svg)](https://pypi.org/project/ncRNABert/) [![GitHub - LICENSE](https://img.shields.io/github/license/wangleiofficial/ncRNABert.svg?style=flat)](./LICENSE) ![PyPI - Downloads](https://img.shields.io/pypi/dm/ncRNABert) [![Wheel](https://img.shields.io/pypi/wheel/ncRNABert)](https://pypi.org/project/ncRNABert/) ![build](https://img.shields.io/github/actions/workflow/status/wangleiofficial/ncRNABert/publish_to_pypi.yml)

## Table of Contents

- [Model Details](#model-details)  
- [Installation](#install)  
- [Usage](#usage)  
  - [ncRNA Sequence Embedding](#ncrna-sequence-embedding)  
- [Comprehensive Benchmarking of Large Language Models](#comprehensive-benchmarking-of-large-language-models)  
- [Zero-Shot Correlation with Experimental Fitness](#zero-shot-correlation-with-experimental-fitness)  
- [License](#license)  

  
### Model details
|   **Model**    | **# of parameters** | **# of hidden size** |            **Pretraining dataset**             | **# of ncRNAs** | **Model download** |
|:--------------:|:-------------------:|:----------------------:|:----------------------------------------------:|:-----------------:|:------------------------:|
|    ncRNABert   |        303M         |           1024           | [RNAcentral](http://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz) |       26M        |      [Download](https://zenodo.org/records/15162985/files/ncRNABert.pt)       |

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
model = load_ncRNABert()
with torch.no_grad():
    results = model(batch_token, lengths, repr_layers=[24])
# Generate per-sequence representations via averaging
token_representations = results["representations"][24]
sequence_representations = []
batch_lens = [len(item[1]) for item in data]
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i].mean(0))
```

### Comprehensive benchmarking of Large Language Models
When comparing the performance of different RNA language models, the ncRNABert model has demonstrated exceptional performance across multiple evaluation metrics. According to the tales, ncRNABert outperforms other models in terms of F1 score, achieving an average accuracy of 0.595, which is the highest among all the models. (https://github.com/sinc-lab/rna-llm-folding)

| Methods    | 16s   | 23s   | 5s    | RNaseP | grp1  | srp   | tRNA  | telomerase | tmRNA | Average |
|------------|-------|-------|-------|--------|-------|-------|-------|------------|-------|---------|
| ERNIE-RNA  | 0.539 | 0.580 | 0.820 | 0.687  | 0.317 | 0.610 | 0.841 | 0.151      | 0.700 | 0.583   |
| RNA-FM     | 0.152 | 0.193 | 0.555 | 0.324  | 0.136 | 0.277 | 0.763 | 0.121      | 0.293 | 0.313   |
| RNA-MSM    | 0.133 | 0.223 | 0.264 | 0.207  | 0.189 | 0.151 | 0.338 | 0.072      | 0.240 | 0.202   |
| RNABERT    | 0.144 | 0.167 | 0.211 | 0.171  | 0.144 | 0.152 | 0.458 | 0.101      | 0.152 | 0.189   |
| RNAErnie   | 0.191 | 0.227 | 0.536 | 0.198  | 0.170 | 0.164 | 0.795 | 0.071      | 0.259 | 0.290   |
| RiNALMo    | 0.473 | 0.596 | 0.796 | 0.667  | 0.566 | 0.548 | 0.845 | 0.093      | 0.669 | 0.584   |
| one-hot    | 0.155 | 0.188 | 0.279 | 0.169  | 0.149 | 0.174 | 0.452 | 0.132      | 0.175 | 0.208   |
| ncRNABert  | 0.573 | 0.733 | 0.773 | 0.629  | 0.423 | 0.589 | 0.789 | 0.161      | 0.688 | 0.595   |


| Methods    | bpRNA | bpRNA-new |
|------------|:-----:|:---------:|
| ERNIE-RNA  | 0.628 | 0.601     |
| RNA-FM     | 0.522 | 0.423     |
| RNA-MSM    | 0.426 | 0.393     |
| RNABERT    | 0.357 | 0.358     |
| RNAErnie   | 0.442 | 0.387     |
| RiNALMo    | 0.599 | 0.446     |
| one-hot    | 0.351 | 0.383     |
| ncRNABert  | 0.595 | 0.572     |



### Zero-Shot Correlation with Experimental Fitness

ncRNABert shows strong zero-shot correlation between language model (pseudo)likelihoods and experimental fitness across seven ncRNA DMS datasets(https://github.com/evo-design/evo/), outperforming other models with an average correlation of 0.294.

| Model                  | Andreasson (2020) | Hayden (2011) | Kobori (2016) | Pitt (2010) | Zhang (2009) | Domingo (2018) | Guy (2014) | Average |
| ---------------------- | ----------------- | ------------- | ------------- | ----------- | ------------ | -------------- | ---------- | ------- |
| Evo                    | 0.14              | 0.13          | 0.17          | 0.14        | 0.60         | 0.45           | 0.24       | 0.267   |
| GenSLM                 | 0.10              | 0.19          | 0.11          | 0.18        | 0.12         | 0.29           | 0.05       | 0.149   |
| Nucleotide Transformer | 0.07              | 0.24          | 0.20          | 0.01        | 0.20         | 0.06           | 0.05       | 0.119   |
| RNA‑FM                 | 0.16              | 0.11          | 0.03          | 0.13        | 0.56         | 0.20           | 0.05       | 0.177   |
| ncRNABert              | 0.14              | 0.24          | 0.18          | 0.01        | 0.69         | 0.43           | 0.37       | 0.294   |

### License
This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of this source tree.
