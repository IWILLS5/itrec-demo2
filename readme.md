# ITREC 新生训练营demo2
该项目基于MSRA和weibo数据集，bert-base-chinese和hfl/chinese-bert-wwm模型, 在原有模型的基础上添加分类头，以完成ner识别的任务。<br>
具体数据集和demo说明参考仓库：**https://github.com/DMU-ITREC/itrec-nlp-newcomer-guide/tree/main**
## 项目文件
- \ datas \ 0.demo1文本分类 \ new_id_convert.json : 新闻id和对应新闻的对应关系，以及新闻id和id的对应关系
- \ config \ base.json : 模型基础配置文件
- datasets_clean.py : 数据预处理文件(清洗以及生成new_id_convert.json)
- datasets.py : 数据集文件
- environment.yml : conda环境文件
- evaluationMetrics.py : 模型指标文件
- model.py : 模型文件
- train.py : 模型训练文件
- utils.py : 通用工具文件

## 快速运行
```
├── ITREC
    ├── datas
    ├   └── 1.demo2实体识别
    ├       ├   MSRA
    ├       └── weibo
    ├── Demo
    ├    └── demo2
    ├       ├── config
    ├       ├   ├   ......
    ├       ├   ├   option.json
    ├       ├   └── base.json
    ├       ├── clean.py
    ├       ├── datasets.py
    ├       ├── environment.yml
    ├       ├── evaluationMetrics.py
    ├       ├── init.py
    ├       ├── model.py
    ├       ├── train.py
    ├       └── utils.py
    ├── outputs
    ├    └── demo2
    ├── pretrained_models
```
运行命令目录(命令均在Demo2路径下):
```
cd ~\ITREC\Demo\Demo2
```
### 1.项目初始化
初始化之前，需要保持以上的项目目录。<br>
在终端输入
```powershell
python init.py
``` 
`*如果超时或者下载较慢，可以提前输入`
```powershell
export HF_ENDPOINT=https://hf-mirror.com
```
1.自动下载模型配置文件<br>
2.获取并打印两个数据集的BIO情况<br>
3.保存两个数据集的class.txt、labels2id.json和labels.json 并 更新options.json文件更新完成
### 2.数据集处理
在终端输入
```powershell
python clean.py
``` 
输入1，
结果如下：<br>
![alt text](md_images/image-1.png)
可以看到在weibo的数据集的train.txt和test.txt文件中的实体BIO标签出现错误，且排查后发现错误都是`实体以I-xxx起始，而不是B-xxx`，如下图所示：
![alt text](md_images/image-2.png)
![alt text](md_images/image-3.png)
接下来继续终端输入
```powershell
python clean.py
``` 
输入2，
结果如下：<br>
![alt text](md_images/image-4.png)

### 3.swanlab登录
```powershell
swanlab login
``` 
登录 swanlab,并修改配置文件中的swanlab_project和swanlab_workspace
```json
"train":{
        "epochs" : 5,
        "ES_patience" : 3,
        "warmup_ration" : 0.1,
        "adamW_bert_lr" : 2e-5,
        "adamW_classifier_lr" : 1e-3,
        "adamW_weight_decay": 0.01,
        "adamW_eps" : 1e-8,
        "seed" : 5,
        "swanlab_project" : "ner-demo2",
        "swanlab_workspace" : "iwills"
    },
```
### 4.运行train.py文件
在终端运行该文件的时候, 根据需要声明自己创建的config文件（configs目录中，默认路径在utils.py中声明）<br>
`*数据集和模型名称必须声明`<br>
可以先-h查看可选数据集和模型
```powershell
python .\train.py -h
usage: train.py [-h] [--config CONFIG] --dataset {MSRA,weibo} --model
                {bert-base-chinese,hfl/chinese-bert-wwm}

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --dataset {MSRA,weibo}
                        数据集名称
  --model {bert-base-chinese,hfl/chinese-bert-wwm}
                        模型名称
```
然后开始训练
```powershell
#使用默认base.json
python train.py --dataset MSRA --model bert-base-chinese
#使用自定义
python train.py --config .\Demo\demo1\configs\example.json --dataset MSRA --model bert-base-chinese
```
## labels2id
### MSRA
labels2id.json
```json
{
    "I-LOC": 0,
    "I-PER": 1,
    "I-ORG": 2,
    "O": 3,
    "B-PER": 4,
    "B-ORG": 5,
    "B-LOC": 6
}
```
labels.json
```json
{
    "train": {
        "B-LOC": 3952,
        "I-LOC": 5313,
        "O": 206412,
        "B-PER": 1850,
        "I-PER": 3612,
        "B-ORG": 2158,
        "I-ORG": 9141
    },
    "dev": {
        "O": 35046,
        "B-LOC": 646,
        "I-LOC": 975,
        "B-PER": 308,
        "I-PER": 559,
        "B-ORG": 325,
        "I-ORG": 1348
    },
    "test": {
        "O": 34967,
        "B-LOC": 648,
        "I-LOC": 1027,
        "B-PER": 510,
        "I-PER": 1082,
        "B-ORG": 287,
        "I-ORG": 1309
    }
}
```
### weibo
labels2id.json
```json
{
    "I-GPE.NAM": 0,
    "B-LOC.NOM": 1,
    "I-GPE.NOM": 2,
    "I-PER.NOM": 3,
    "B-LOC.NAM": 4,
    "I-PER.NAM": 5,
    "I-LOC.NOM": 6,
    "I-ORG.NAM": 7,
    "O": 8,
    "B-GPE.NAM": 9,
    "B-ORG.NOM": 10,
    "B-GPE.NOM": 11,
    "B-PER.NOM": 12,
    "I-LOC.NAM": 13,
    "B-PER.NAM": 14,
    "I-ORG.NOM": 15,
    "B-ORG.NAM": 16
}
```
labels.json
```json
{
    "train": {
        "O": 68777,
        "B-PER.NOM": 772,
        "I-PER.NOM": 1037,
        "B-LOC.NAM": 56,
        "I-LOC.NAM": 129,
        "B-PER.NAM": 578,
        "I-PER.NAM": 1037,
        "B-GPE.NAM": 205,
        "I-GPE.NAM": 241,
        "B-ORG.NAM": 183,
        "I-ORG.NAM": 477,
        "B-ORG.NOM": 42,
        "I-ORG.NOM": 61,
        "B-LOC.NOM": 51,
        "I-LOC.NOM": 66,
        "B-GPE.NOM": 8,
        "I-GPE.NOM": 8
    },
    "dev": {
        "O": 13492,
        "B-PER.NOM": 208,
        "I-PER.NOM": 239,
        "B-GPE.NAM": 26,
        "I-GPE.NAM": 31,
        "B-ORG.NOM": 5,
        "I-ORG.NOM": 5,
        "B-LOC.NOM": 6,
        "I-LOC.NOM": 7,
        "B-PER.NAM": 90,
        "I-PER.NAM": 157,
        "B-ORG.NAM": 47,
        "I-ORG.NAM": 126,
        "B-LOC.NAM": 6,
        "I-LOC.NAM": 16,
        "B-GPE.NOM": 1,
        "I-GPE.NOM": 1
    },
    "test": {
        "O": 13732,
        "B-PER.NAM": 113,
        "I-PER.NAM": 200,
        "B-GPE.NAM": 47,
        "I-GPE.NAM": 62,
        "B-ORG.NOM": 17,
        "I-ORG.NOM": 22,
        "B-PER.NOM": 172,
        "I-PER.NOM": 214,
        "B-LOC.NAM": 19,
        "I-LOC.NAM": 45,
        "B-LOC.NOM": 9,
        "I-LOC.NOM": 15,
        "B-GPE.NOM": 2,
        "I-GPE.NOM": 2,
        "B-ORG.NAM": 39,
        "I-ORG.NAM": 100
    }
}
```