import torch.nn as nn
from transformers import AutoModel

class model(nn.Module):
    def __init__(self, config):
        super().__init__()

        pt_model_path = config['pt_model_path']
        num_labels = config['num_labels']
        dropout = config['dropout']

        self.pre_trained_model = AutoModel.from_pretrained(pt_model_path)
        

if __name__ == "__main__":
    pass