import torch
from torch.utils.data import Dataset, DataLoader

class Datasets(Dataset):
    def __init__(self, data_path, tokenizer, labels2id, ignore_index, max_len = 512):
        super().__init__()

        self.tokenizer = tokenizer
        self.texts, self.labels = [], []
        self.max_len = max_len
        self.labels2id = labels2id
        self.ignore_index = ignore_index

        with open(data_path, 'r', encoding='utf-8') as f:
            text, label = [], []
            for line in f.readlines():
                #遇到空行，即一个预料完成
                if not line.strip():
                    if text:
                        self.texts.append(text)
                        self.labels.append(label)
                    text, label = [], []
                else:
                    ss_line = line.strip().split()
                    text.append(ss_line[0])
                    label.append(labels2id[ss_line[1]])
            if text:
                self.texts.append(text)
                self.labels.append(label)
 
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def collate_fn(self, batch):
        texts, labels = [], []
        for data in batch:
            texts.append(data[0])
            labels.append(data[1])
        tokens = self.tokenizer(
            texts,
            is_split_into_words=True,
            max_length = self.max_len,
            padding = 'longest',
            truncation = True,
            return_tensors = 'pt'
        )
        #* 以下是标签对齐操作
        labels_id = []
        #这个labels已经是id，只不过没有对齐
        for i, label in enumerate(labels):
            #获取对齐后的内容（仅包含None和在label中的下标）
            wrod_id = tokens.word_ids(batch_index = i)
            #将None转化为-100和将label下标转化为id
            label_id = []
            
            for id in wrod_id:
                if id is None:
                    label_id.append(self.ignore_index)
                else:
                    label_id.append(label[id])
            labels_id.append(label_id)

        labels_id = torch.tensor(labels_id, dtype=torch.long)
        tokens["labels_id"] = labels_id
        
        return tokens
    def get_dataLoader(self, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            #* 如果不使用自定义的collate，默认的collate会检测到样本长度不一致，而报错
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


if __name__ == "__main__":
    import json
    from transformers import AutoTokenizer
    from utils import args_parse
    from model import Model
    from evaluationMetrics import EvaluationMetrics

    config, dataset_n, model_n = args_parse()
    data_config = config['data']
    with open(data_config['datasets']['labels2id'], 'r', encoding='utf-8') as f:
        labels2id = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pt_model_path'])
    

    dev_datas = Datasets(data_config['datasets']['dev'], tokenizer, labels2id, data_config["ignore_index"])
    #dataloader的参数
    batch_size = data_config['batch_size']
    shuffle = True
    num_workers = data_config['num_workers']
    pin_memory = True if torch.cuda.is_available() else False
    dataloader = dev_datas.get_dataLoader(batch_size, shuffle, num_workers, pin_memory)
    #模型配置
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = Model(config['model']).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=data_config['ignore_index'])
    optimizer = torch.optim.AdamW([
        {'params':model.pre_trained_model.parameters(), 'lr':config['train']['adamW_bert_lr']},
        {'params':model.nerClassifier.parameters(), 'lr':config['train']['adamW_classifier_lr']}
    ], weight_decay=0.01, eps=1e-8)


    id2labels = {
        value : key for key, value in labels2id.items()
    }

    em = EvaluationMetrics(id2labels)
    for batch in dataloader:
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


        print(f"{output.argmax(-1) = }")
        print(f"{labels_id = }")

        print(f"{output_res.argmax(-1) = }")
        print(f"{labels_id_res = }")

        # valid_mask = labels_id_res != -100
        for i, label_id in enumerate(labels_id):
            print(f"-----------------------------{i}-----------------------------")
            valid_mask = label_id != data_config['ignore_index']
            print(f"{label_id[valid_mask].tolist() = }")
            print(f"{output[i].argmax(dim=-1)[valid_mask].tolist() = }")
        # print(f"{output_res.argmax(-1) = }")
        # print(f"{labels_id_res = }")

        # print(f"{output_res.argmax(-1)[valid_mask] = }")
        # print(f"{labels_id_res[valid_mask] = }")
        break
    
    