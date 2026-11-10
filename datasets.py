import torch
from torch.utils.data import Dataset, DataLoader

class Datasets(Dataset):
    def __init__(self, data_path, tokenizer, labels2id, max_len = 512):
        super().__init__()

        self.tokenizer = tokenizer
        self.texts, self.labels = [], []
        self.max_len = max_len
        self.labels2id = labels2id

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
        print(f'{batch = }')
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
                    label_id.append(-100)
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

    inputs = ['大连海事大学', '我是曹超森']

    config = args_parse()
    data_config = config['data']
    with open(data_config['datasets']['labels2id'], 'r', encoding='utf-8') as f:
        labels2id = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pt_model_path'])
    

    dev_datas = Datasets(data_config['datasets']['dev'], tokenizer, labels2id)
    # print(len(dev_datas))
    batch_size = data_config['batch_size']
    shuffle = True
    num_workers = data_config['num_workers']
    pin_memory = True if torch.cuda.is_available() else False
    dataloader = dev_datas.get_dataLoader(batch_size, shuffle, num_workers, pin_memory)
    a = 0
    for batch in dataloader:
        a += 1
        print(f"输出{batch}")
        if a >= 1:
            break
    