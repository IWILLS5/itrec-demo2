import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer 
import swanlab

from utils import args_parse, op_parse
from model import Model
from datasets import Datasets
from evaluationMetrics import EvaluationMetrics


class Experiment:
    def __init__(self, config, labels2id, dataset_n, model_n):
        self.config = config
        self.dataset_n = dataset_n
        self.model_n = model_n
        self.labels2id = labels2id
        self.id2labels = {
            value : key for key, value in labels2id.items()
        }
        #指标对象
        self.train_em = EvaluationMetrics(self.id2labels)
        self.evaluate_em = EvaluationMetrics(self.id2labels)

        self.ignore_index = config['data']['ignore_index']

        #config['train'] 参数
        self.epochs = self.config['train']['epochs']
        self.EarlyStop_patience = self.config['train']['ES_patience'] #Early Stop的patience
        self.warmup_ration = self.config['train']['warmup_ration']
        self.adamW_bert_lr = self.config['train']['adamW_bert_lr']
        self.adamW_nerClassifier_lr = self.config['train']['adamW_classifier_lr']
        self.adamW_weight_decay = self.config['train']['adamW_weight_decay']
        self.adamW_eps = self.config['train']['adamW_eps']
        self.swanlab_project = config['train']['swanlab_project']
        self.swanlab_workspace = self.config['train']['swanlab_workspace']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(config['model']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['pt_model_path'])
        self.datasets_len, self.dataLoaders = self.get_DataLoader(
            self.config['data'], self.tokenizer, labels2id
        )
        #交叉熵损失+softmax（忽略index=-100）
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.ignore_index)
        
        #optimizer 和 scheduler
        self.optimizer = torch.optim.AdamW([
            {'params':self.model.pre_trained_model.parameters(), 'lr':self.adamW_bert_lr},
            {'params':self.model.nerClassifier.parameters(), 'lr':self.adamW_nerClassifier_lr}
        ], weight_decay=self.adamW_weight_decay, eps=self.adamW_eps)
        #获取总共的batch次数
        total_steps = self.epochs * len(self.dataLoaders['train'])
        warmup_steps = max(1, int(total_steps * self.warmup_ration))
        decay_steps = max(total_steps - warmup_steps, 1)
        #由base_lr 升到 lr，步数是warmup_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        #从lr降至最后学习率eta_min的步数是decay_steps
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps,
            eta_min=0
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )

        output_folder = Path(config['exp']['op_dir'])
        output_path = op_parse(output_folder, self.config, self.dataset_n, self.model_n)
        if os.path.exists(output_path):
            option = input("实验数据文件夹已存在，请确认已经保存，是否继续(y or n):\n")
            if option not in ('y', 'yes'):
                raise SystemExit("已终止程序运行！！！")
        else:
            os.mkdir(output_path)

        self.best_check_point_path = output_path / 'best_model.pt'
        self.history_path = output_path / 'history.json'
        self.result_path = output_path / 'result.json'

    #保存历史数据
    def save_history(self, history, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    #保存最优checkpoint
    def _save_best_checkpoint(self, epoch, best_entity_f1):
            
        checkpoint = {
            'dataset' : self.dataset_n,
            'model' : self.model_n,
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'labels2id' : self.labels2id,
            'dev_entity_f1' : best_entity_f1
        }
        torch.save(checkpoint, self.best_check_point_path)

    # 训练一个epoch
    def train_epoch(self, model, dataLoader, criterion, 
                    optimizer, scheduler, evaluationMatrics : EvaluationMetrics, device):
        #train模式
        model.train()

        epoch_loss = 0.0 
        correct = 0
        samples = 0

        for batch in dataLoader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_id = batch['labels_id'].to(device)

            labels_id_res = labels_id.reshape(-1)
            #梯度清零和计算
            optimizer.zero_grad()
            output = model(input_ids, token_type_ids, attention_mask)
            #将output降维，0和1维展平
            output_res = output.reshape(-1, output.size(-1))
            loss = criterion(output_res, labels_id_res)
            loss.backward()
            #梯度下降和shceduler更新
            optimizer.step()
            scheduler.step()

            #指标类传参(按文本传参)
            for i, label_id in enumerate(labels_id):
                valid_mask = label_id != self.ignore_index
                evaluationMatrics.add_batch(
                    label_id[valid_mask].tolist(),
                    output[i].argmax(dim=-1)[valid_mask].tolist()
                )

            valid_mask = labels_id_res != self.ignore_index
            batch_token_size = len(labels_id_res[valid_mask])

            epoch_loss += loss.item() * batch_token_size
            correct += (labels_id_res[valid_mask] == 
                            output_res.argmax(dim=-1)[valid_mask]).sum().item()
            samples += batch_token_size

        #返回一个epoch的loss和token准确率
        return epoch_loss / samples, correct / samples
    
    #进行dev和test的评估
    @torch.inference_mode() #比torch.no_grad()更强
    def evaluate(self, model, dataLoader, criterion, evaluationMatrics, device):
        model.eval()

        epoch_loss = 0.0
        correct = 0
        samples = 0

        for batch in dataLoader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_id = batch['labels_id'].to(device)

            labels_id_res = labels_id.reshape(-1)

            output = model(input_ids, token_type_ids, attention_mask)
            #将output降维，0和1维展平
            output_res = output.reshape(-1, output.size(-1))
            loss = criterion(output_res, labels_id_res)

            #指标类传参(按文本传参)
            for i, label_id in enumerate(labels_id):
                valid_mask = label_id != self.ignore_index
                evaluationMatrics.add_batch(
                    label_id[valid_mask].tolist(),
                    output[i].argmax(dim=-1)[valid_mask].tolist()
                )

            valid_mask = labels_id_res != self.ignore_index
            batch_token_size = len(labels_id_res[valid_mask])

            epoch_loss += loss.item() * batch_token_size
            correct += (labels_id_res[valid_mask] == 
                            output_res.argmax(dim=-1)[valid_mask]).sum().item()
            samples += batch_token_size

        #返回batch的平均loss和准确率
        return epoch_loss / samples, correct / samples
        
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
                data_path = path, tokenizer = tokenizer, labels2id = labels2id, ignore_index = self.ignore_index, max_len = config['max_len']
            )
            datasets[name] = dataset
            datasets_len[name] = len(dataset)
        #构建dataLoader
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        pin_memory = self.device.type == 'cuda'

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
            'train_token_loss' : [],
            'train_token_acc' : [],
            'dev_token_loss' : [],
            'dev_token_acc' : [],
            'train_entity_precision' : [],
            'train_entity_recall' : [],
            'train_entity_f1' : [],
            'dev_entity_precision' : [],
            'dev_entity_recall' : [],
            'dev_entity_f1' : []
        }
        
        best_entity_f1, best_token_loss = float('-inf'), float('-inf')
        count_bad_epochs = 0

        print('----------------配置展示----------------')
        print(f'运行设备:{self.device}')
        print(f'使用数据集:{self.dataset_n},使用模型:{self.model_n}')
        for name, size in self.datasets_len.items():
            print(f'{name}数据集的长度={size}')

        for epoch in range(1, self.epochs + 1):
            #指标清零
            self.train_em.reset()
            train_token_loss, train_token_acc = self.train_epoch(
                self.model, 
                self.dataLoaders['train'], 
                self.criterion, self.optimizer,
                self.scheduler ,self.train_em, 
                self.device
            )
            train_entity_precision, train_entity_recall, train_entity_f1 = self.train_em.get_Micro_data()
            
            self.evaluate_em.reset()
            dev_token_loss, dev_token_acc = self.evaluate(
                self.model, 
                self.dataLoaders['dev'], 
                self.criterion,
                self.evaluate_em, 
                self.device
            )
            dev_entity_precision, dev_entity_recall, dev_entity_f1 = self.evaluate_em.get_Micro_data()
            
            history['train_token_loss'].append(train_token_loss)
            history['train_token_acc'].append(train_token_acc)
            history['train_entity_precision'].append(train_entity_precision)
            history['train_entity_recall'].append(train_entity_recall)
            history['train_entity_f1'].append(train_entity_f1)
            history['dev_token_loss'].append(dev_token_loss)
            history['dev_token_acc'].append(dev_token_acc)
            history['dev_entity_precision'].append(dev_entity_precision)
            history['dev_entity_recall'].append(dev_entity_recall)
            history['dev_entity_f1'].append(dev_entity_f1)

            swanlab.log({
                'train_token_loss' : train_token_loss,
                'train_token_acc' : train_token_acc,
                'train_entity_precision' : train_entity_precision,
                'train_entity_recall' : train_entity_recall,
                'train_entity_f1' : train_entity_f1,
                'dev_token_loss' : dev_token_loss,
                'dev_token_acc' : dev_token_acc,
                'dev_entity_precision' : dev_entity_precision,
                'dev_entity_recall' : dev_entity_recall,
                'dev_entity_f1' : dev_entity_f1
            })

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"train_token_loss : {train_token_loss:.4f} | "
                f"train_token_acc : {train_token_acc:.4f} | "
                f"train_entity_f1 : {train_entity_f1:.4f} | "
                f"dev_token_loss : {dev_token_loss:.4f} | "
                f"dev_token_acc : {dev_token_acc:.4f} | "
                f"dev_entity_f1 : {dev_entity_f1:.4f}"
            )

            #检测模型指标是否提升
            if best_entity_f1 < dev_entity_f1:
                best_entity_f1 = dev_entity_f1
                best_token_loss = dev_token_loss
                self._save_best_checkpoint(epoch, best_entity_f1)
                count_bad_epochs = 0
            else:
                if best_entity_f1 == dev_entity_f1 and dev_token_loss < best_token_loss:
                    self._save_best_checkpoint(epoch, best_entity_f1)
                    best_token_loss = dev_token_loss
                count_bad_epochs += 1
            #检测early stopping
            if count_bad_epochs >= self.EarlyStop_patience:
                print(f"Early Stopping at epoch {epoch}")
                break
        #保存历史记录
        self.save_history(history, self.history_path)


    def test_best_model(self):
        checkpoint = torch.load(
            self.best_check_point_path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        #先将指标内容清零
        self.evaluate_em.reset()
        test_token_loss, test_token_acc = self.evaluate(
            self.model,
            self.dataLoaders['test'],
            self.criterion,
            self.evaluate_em,
            self.device
        )

        entity_precision, entity_recall, entity_f1 = self.evaluate_em.get_Micro_data()

        result = {
            "best_epoch" : int(checkpoint['epoch']),
            "best_entity_f1" : float(checkpoint['dev_entity_f1']),
            "test_token_loss" : test_token_loss,
            "test_token_acc" : test_token_acc,
            "entity_precision" : entity_precision,
            "entity_recall" : entity_recall,
            "entity_f1" : entity_f1,
        }

        with open(self.result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent = 2)
        print(
            f"Test Token Loss: {test_token_loss:.4f} | Entity_f1: {entity_f1:.4f} "
            f"(best epoch: {result['best_epoch']})"
        )
        # resolve()打印绝对路径
        print(f"最佳模型已保存至：{self.best_check_point_path.resolve()}")


    def run(self):
        self.train()
        self.test_best_model()


def main():
    config, dataset_n, model_n = args_parse()
    with open(config['data']['datasets']['labels2id'], 'r', encoding='utf-8') as f:
        labels2id = json.load(f)
    exp = Experiment(config, labels2id, dataset_n, model_n)
    exp.run()



if __name__ == "__main__":
    main()
