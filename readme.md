# ITREC 新生训练营demo2
## 🧪 Demo 2：命名实体识别（NER）任务

### ✅ 任务目标

分别使用下面二个bert预训练模型[bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)和[chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm)，完成weibo和msra二个数据集中文实体识别任务，理解序列标注模型的构建逻辑与评估指标。

### 🔧 实现要求

1. 使用 PyTorch 和 Hugging Face 生态，**独立完成以下模块**：
   * 代码需要同时支持两种bert预训练模型和数据集
   * 数据加载与预处理（BIO标注格式）
   * 模型设计与训练流程
   * 模型评估与结果可视化 
3. 使用 [SwanLab](https://www.swanlab.cn/) 等工具进行训练过程与结果的可视化。
4. 使用**微博实体识别数据集和MSRA实体识别数据集**，详见网盘数据集中的demo2实体识别，[微博数据来源参考](https://aclanthology.org/D15-1064.pdf)、[MSRA数据来源参考](https://tianchi.aliyun.com/dataset/144307)。

### 📊 实验汇报要求

* 参数调优对实体识别性能的变化趋势