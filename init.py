import os
import json
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

#初始化保存model及tokenizer配置
def init_model_tokenizer(config, save_models_path):
    
    options = list(config['models'].keys())

    for model_p in options:
        path = f'{save_models_path}/{model_p}'
        if not os.path.isdir(path):
            model = AutoModel.from_pretrained(model_p)
            tokenizer = AutoTokenizer.from_pretrained(model_p)
            
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
        else:
            print(f"{model_p}的配置已保存!")
#查看各个数据集的每个文件的标签分布 及 保存文件
def viewLabels(config):

    datasets_config = config['datasets'] 
    datasets = list(datasets_config.keys()) 
    for dataset in datasets:
        #获取train、dev、test路径，去除class文件
        data_path = list(datasets_config[dataset].keys())
        #遍历各个
        print(f"{dataset}的标签数量分布如下:")
        labels = {}
        for da_p in data_path:
            #跳过class
            if da_p == 'class' or da_p == 'labels2id':
                continue

            class_cnt = {}
            with open(datasets_config[dataset][da_p], 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    #防止空行
                    if data:
                        #统计labels
                        #统计当前预料的labels数量
                        if class_cnt.get(data[1], 0) != 0:
                            class_cnt[data[1]] += 1
                        else:
                            class_cnt[data[1]] = 1
            labels[da_p] = class_cnt

        print(labels)
        path = Path(config['datasets'][dataset]['train']).parent / 'labels.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=4, ensure_ascii=False)
        print(f"保存到文件labels.json, 路径={path}")

#遍历数据集，然后获取标签类型
def getLabels(config, options_path):
    datasets_config = config['datasets']
    #数据集名称
    datasets = list(datasets_config.keys())
    for dataset in datasets:
        #查看是否含有class.txt文件
        # if datasets_config[dataset].get('class', None) == None:
            #获取train、dev、test路径
        data_path = list(datasets_config[dataset].keys())
        if 'class' in set(data_path):
            data_path.remove('class')
        if 'labels2id' in set(data_path):
            data_path.remove('labels2id')
        #遍历各个
        labels = [set() for _ in range(len(data_path))]
        for i, da_p in enumerate(data_path):
            with open(datasets_config[dataset][da_p], 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    #防止空行
                    if data:
                        #统计labels
                        if data[1] not in labels[i]:
                            labels[i].add(data[1])
            print(f"{da_p} = {labels[i]}")
        #判断所有文件的标签都一致
        label = labels[0]
        for i in range(1, len(labels)):
            if label != labels[i]:
                print("数据集之间含有不同的labels")

        #class.txt
        class_path = Path(datasets_config[dataset]['train']).parent / "class.txt"
        #写入class.txt文件
        with open(class_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(lab for lab in label))
        print("class.txt文件已写入完成")

        #写入labels2id.json文件
        labels2id_path = Path(datasets_config[dataset]['train']).parent / "labels2id.json"
        labels2id = {}
        for i, lab in enumerate(label):
            labels2id[lab] = i
        with open(labels2id_path, 'w', encoding='utf-8') as f:
            json.dump(labels2id, f, indent=4, ensure_ascii=False)
        print("labels2id.json文件已写入完成")

        #更新options文件
        config['datasets'][dataset]['class'] = str(class_path)
        config['datasets'][dataset]['labels2id'] = str(labels2id_path)
        with open(options_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("options.json文件更新完成")



if __name__ == "__main__":
    save_models_path = "../../pretrained_models"
    options_path = './config/options.json'
    with open(options_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print("--------------------model和tokenizer初始化...--------------------")
    init_model_tokenizer(config, save_models_path)
    print("--------------------数据集labels获取并保存中...--------------------")
    getLabels(config, options_path)
    print("--------------------数据集labels分布获取并保存中...--------------------")
    viewLabels(config)