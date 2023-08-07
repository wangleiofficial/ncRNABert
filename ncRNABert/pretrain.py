""" 
"""
import torch
from .model import RNABertModel
from .utils import load_hub_workaround, token_to_index

MODEL_URL = ""


def load_ncRNANert():
    model_data = load_hub_workaround(MODEL_URL)
    hyper_parameter = model_data["hyper_parameters"]
    model = RNABertModel(num_token=hyper_parameter['num_token'], num_layers=hyper_parameter['num_layers'],
                         embed_dim=hyper_parameter['hyper_parameter'], attention_heads=hyper_parameter['attention_heads'],
                         padding_idx=token_to_index['PADDING_MASK'])

    model.load_state_dict(model_data['state_dict'])

    return model

if __name__ == "__main__":
    model = load_ncRNANert()
