import json
import argparse

import torch

#数据集和模型配置
options_path = "./config/options.json"
#模型默认配置
default_config_path = "./config/base.json"

def load_option():
    #导入option
    with open(options_path, 'r', encoding="utf-8") as f:
        return json.load(f)


def args_parse():
    options = load_option()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=default_config_path
    )
    parser.add_argument(
        "--dataset", choices=options['datasets'], help="数据集名称", required=True
    )
    parser.add_argument(
        "--model", choices=options['models'], help="模型名称", required=True
    )
    args = parser.parse_args()
    config, dataset, model = args.config, args.dataset, args.model
    #导入config
    try:
        with open(config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件不存在：{config}")
        option = input("是否选择继续使用base.json配置(y or n):\n")
        if option == 'y':
            with open(default_config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
        else:
            raise

    config['data']['datasets'] = options['datasets'][dataset]
    config['model']['pt_model_path'] = options['models'][model]

    with open(config['data']['datasets']['labels2id'], 'r', encoding='utf-8') as f:
        config['model']['num_labels'] = len(json.load(f))
    
    return config, dataset, model


def fmt(x):
    if isinstance(x, int):
        return str(x)
    s = f"{x:e}"                       # 1e-3 → '1.000000e-03'
    mantissa, exp = s.split('e')
    mantissa = mantissa.rstrip('0.')  # '1.000000' → '1'
    exp = str(int(exp))                # '-03' → '-3'
    return f"{mantissa}e{exp}"

def op_parse(output_folder, config, dataset_n, model_n):
    
    model_splits = model_n.split('-')
    model_n = "".join([model_split[0] for model_split in model_splits])
    
    parts = [
        str(dataset_n),
        str(model_n),
        str(config['train']['epochs']), 
        str(config['data']['batch_size']),
        str(config['model']['dropout']),
        fmt(config['train']['adamW_bert_lr']),
        fmt(config['train']['adamW_classifier_lr'])
    ]

    return output_folder/ '_'.join(parts)


def seed_init(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print(args_parse())
