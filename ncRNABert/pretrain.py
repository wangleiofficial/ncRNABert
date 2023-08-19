""" 
"""
import torch
from .model import RNABertModel
from .utils import load_hub_workaround, token_to_index

MODEL_URL = "https://zenodo.org/record/8263889/files/ncRNABert.pt"


def load_ncRNABert():
    model_data = load_hub_workaround(MODEL_URL)
    # model_data = torch.load("/home/wanglei/data/RNALM/src/ncRNABert.pt")
    hyper_parameter = model_data["hyper_parameters"]
    model = RNABertModel(num_token=hyper_parameter['num_tokens'],
                         num_layers=hyper_parameter['num_layer'],
                         embed_dim=hyper_parameter['embed_dim'],
                         attention_heads=hyper_parameter['attention_heads'],
                         dropout=hyper_parameter['dropout'],
                         padding_idx=token_to_index["PADDING_MASK"])

    model.load_state_dict(model_data['state_dict'])

    return model


if __name__ == "__main__":
    model = load_ncRNABert()
