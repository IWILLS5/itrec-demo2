import json
import argparse

#数据集和模型配置
options_path = "./config/options.json"
#模型默认配置
default_config_path = "./config/base.json"

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=default_config_path
    )
    args = parser.parse_args()

    #导入config
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件不存在：{args.config}")
        option = input("是否选择继续使用base.json配置(y or n):\n")
        if option == 'y':
            with open(default_config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
        else:
            raise
    #导入option
    with open(options_path, 'r', encoding="utf-8") as f:
        options = json.load(f)
    
    datasets, models = options['datasets'].keys(), options['models'].keys()
    datasets_len, models_len = len(datasets), len(models)
    
    print("数据集列表如下:")
    for i, datasets_n in enumerate(datasets):
        print(f"{i}.{datasets_n}")
    datasets_option = int(input("请选择使用的数据集(序号):"))
    while not 0 <= datasets_option < datasets_len:
        print("序号不在范围内!!!")
        datasets_option = int(input("请选择使用的数据集(序号):"))
    for i, datasets_n in enumerate(datasets):
        if i == datasets_option:
            dataset = datasets_n        
    config['data']['datasets'] = options['datasets'][dataset]
    
    print("模型列表如下:")
    for i, model_n in enumerate(models):
        print(f"{i}.{model_n}")
    model_option = int(input("请选择使用的模型(序号):"))
    while not 0 <= model_option < models_len:
        print("序号不在范围内!!!")
        model_option = int(input("请选择使用的模型(序号):"))
    for i, model_n in enumerate(models):
        if i == datasets_option:
            model = model_n  
    config['model']['pt_model_path'] = options['models'][model]
    return config


if __name__ == "__main__":
    print(args_parse())