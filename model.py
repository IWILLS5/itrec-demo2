import torch
import torch.nn as nn
from transformers import AutoModel

from utils import args_parse

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        pt_model_path = config['pt_model_path']
        num_labels = config['num_labels']
        dropout = config['dropout']

        self.pre_trained_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path = pt_model_path, add_pooling_layer = False
        )
        #取出模型token的维度
        self.hidden_size = self.pre_trained_model.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.nerClassifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.pre_trained_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        #去除base模型输出的数据（batch、seq、hidden_size）
        output = self.dropout(output.last_hidden_state)
        logits = self.nerClassifier(output)
        return logits

if __name__ == "__main__":
    config, dataset_n, model_n = args_parse()
    model = Model(config['model'])
    # print(model.hidden_size)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pt_model_path'])
    
    inputs = ['大连海事大学', '我是曹超森']
    tokens = tokenizer(
        inputs,
        # is_split_into_words = True,
        max_length = 512,
        padding = 'longest',
        truncation = True,
        return_tensors = 'pt'
        )
    print(tokens)
    output = model(tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask'])

    print(output)