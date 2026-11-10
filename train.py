import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer 
import swanlab

from utils import args_parse
from model import Model
from datasets import Datasets

class Experiment:
    def __init__(self, config, labels2id, dataset_n, model_n):
        self.config = config
        self.dataset_n = dataset_n
        self.model_n = model_n


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(config['model']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['pt_model_path'])
        self.datasets_len, self.dataLoaders = self.get_DataLoader(
            self.config['data'], self.tokenizer, labels2id
        )
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = self.config['train']['epochs']

        self.swanlab_project = config['train']['swanlab_project']
        self.swanlab_workspace = self.config['train']['swanlab_workspace']

    #保存历史数据
    def save_history(self, history, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    #保存最优checkpoint
    def _save_best_checkpoint(self, epoch, dev_acc):
            
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'class_id2index' : self.new_id_convert,
            'dev_acc' : dev_acc
        }
        torch.save(checkpoint, self.best_check_point_path)

    # 训练一个epoch
    def train_epoch(self, model, dataLoader, criterion, optimizer, scheduler, device):
        
        model.train()
        training_loss = 0.0
        training_correct = 0
        training_samples = 0

        for batch in dataLoader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_id = batch['labels_id'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(output, labels_id)
            loss.backward()

            optimizer.step()
            scheduler.step()

            batch_size = labels_id.size(0)
            training_loss += loss.item() * batch_size
            training_correct += (output.argmax(dim=1) == labels_id).sum().item()
            training_samples += batch_size

        #返回batch的平均loss和准确率
        return training_loss / training_samples, training_correct / training_samples


    def get_DataLoader(self, config, tokenizer, labels2id):
        #获取数据
        datasets, datasets_len = {}, {}
        datasets_dit = {
            'train' : config['datasets']['train'],
            'dev' : config['datasets']['dev'],
            'test' : config['datasets']['test']
        }
        #构建数据集
        for name, path in datasets_dit.items():
            dataset = Datasets(
                data_path = path, tokenizer = tokenizer, labels2id = labels2id, max_len = config['max_len']
            )
            datasets[name] = dataset
            datasets_len[name] = len(dataset)
        #构建dataLoader
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        pin_memory = True if self.device == 'cuda' else False

        dataLoaders = {}
        for name, dataset in datasets.items():
            shuffle = False
            if name == 'train':
                shuffle = True
            dataLoaders[name] = dataset.get_dataLoader(
                batch_size, shuffle, num_workers, pin_memory
            )
        return datasets_len, dataLoaders

    def train(self):


        # 初始化一个新的 swanlab 实验
        swanlab.init(
            # 设置将记录此次实验的项目信息
            project=self.config['train']['swanlab_project'],
            workspace=self.config['train']['swanlab_workspace'],
            # 跟踪超参数和实验元数据
            config={
                "dataset" : self.dataset_n,
                "model" : self.model_n,
                "adamW_bert_lr": self.config['train']['adamW_bert_lr'],
                "adamW_classifier_lr": self.config['train']['adamW_classifier_lr'],
                "epochs": self.epochs,
                "dropout" : self.config['model']['dropout'],
                "batch_size" : self.config['data']['batch_size'],
            }
        )

        history = {
            'train_loss' : [],
            'train_acc' : [],
            'dev_loss' : [],
            'dev_acc' : [],
            'dev_precision' : [],
            'dev_recall' : [],
            'dev_f1' : []
        }
        

        best_dev_acc, best_dev_loss = float('-inf'), float('inf')
        count_bad_epochs = 0

        print('----------------配置展示----------------')
        print(f'运行设备:{self.device}')
        print(f'使用数据集:{self.dataset_n},使用模型:{self.model_n}')
        for name, size in self.datasets_len.items():
            print(f'{name}数据集的长度={size}')

    def run(self):
        self.train()


def main():
    config, dataset_n, model_n = args_parse()
    with open(config['data']['datasets']['labels2id'], 'r', encoding='utf-8') as f:
        labels2id = json.load(f)
    exp = Experiment(config, labels2id, dataset_n, model_n)
    exp.run()



if __name__ == "__main__":
    main()