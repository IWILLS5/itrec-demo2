import json
from transformers import AutoModel, AutoTokenizer

#初始化保存model及tokenizer配置
def init_model_tokenizer(config):
    options = list(config['models'].keys())
    options[1] = "hfl/chinese-bert-wwm"
    models = [AutoModel.from_pretrained(model_p) for model_p in options]
    tokenizer = [AutoTokenizer.from_pretrained(model_p) for model_p in options]
    print(models)
    print(tokenizer)





if __name__ == "__main__":
    with open('./config/options.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    init_model_tokenizer(config)