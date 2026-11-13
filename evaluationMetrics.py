class EvaluationMetrics():
    def __init__(self, id2labels):
        self.id2labels = id2labels
        
        self.TP = 0 
        self.T_sort = 0 #真实是实体
        self.P_sort = 0 #预测为实体

        self.precision = 0
        self.recall = 0
        self.F1 = 0

    def reset(self):
        self.TP = 0
        self.T_sort = 0 #真实是实体
        self.P_sort = 0 #预测为实体
        self.precision = 0
        self.recall = 0
        self.F1 = 0

    #添加一个batch的数据
    def add_batch(self, trues, predictions):
        t_entitys, p_entitys = [], []
        t_entity_flag, p_entity_flag = False, False
        t_entity_end, p_entity_end = "", "" # 实体的结尾
        t_entity, p_entity = [], []
        for i, data in enumerate(zip(trues, predictions)):
            t, p = self.id2labels[data[0]], self.id2labels[data[1]]
            #* 分别统计t和p的实体以及边界，因为T一定是正确的标签，所以碰到B开头，
            #* 后续一定为该类型的I或者O，所以不需要做额外的判断，但是P不一定

            ## 记录格式[PER, 0, 2] : [BIO标签，start, end]
            #如果真实BIO标签是“B-xxx”,开始记录
            if t.startswith("B"):
                #如果遇到连续"B-xxx"标签
                if t_entity_flag:
                    t_entitys.append(t_entity)
                    t_entity = []
                t_entity_end = t.split('-', 1)[1]

                t_entity = [t_entity_end, i, i]
                t_entity_flag = True
            elif t.startswith("I"):
                if t_entity_flag:
                    t_entity[2] = i
            #如果真实BIO标签是“O”
            elif t == "O":
                #如果t_entity_flag是True，那么实体范围刚结束
                if t_entity_flag:
                    t_entity_flag = False
                    #实体结尾记录
                    t_entity[2] = i - 1
                    #实体空间添加
                    t_entitys.append(t_entity)
                    t_entity = []
            
            if p.startswith("B"):
                if p_entity_flag:
                    p_entitys.append(p_entity)
                    p_entity = []
                #记录结尾，T不用记录
                p_entity_end = p.split('-', 1)[1]
                p_entity = [p_entity_end, i, i]
                p_entity_flag = True
            elif p.startswith("I"):
                #如果实体记录开始
                if p_entity_flag:
                    #如果为一个实体，更新右边界
                    if p.split('-')[1] == p_entity_end:
                        p_entity[2] = i
                    #如果不是一个，直接更新实体空间
                    else:
                        p_entitys.append(p_entity)
                        p_entity = []
                        p_entity_flag = False
            elif p == "O":
                if p_entity_flag:
                    p_entity[2] = i - 1
                    p_entitys.append(p_entity)
                    p_entity = []
                    p_entity_flag = False
            
        if t_entity:
            t_entitys.append(t_entity)
        if p_entity:
            p_entitys.append(p_entity)
        #记录数据
        self.T_sort += len(t_entitys)
        self.P_sort += len(p_entitys)
        t_entitys_set = {tuple(t_entity) for t_entity in t_entitys}
        p_entitys_set = {tuple(p_entity) for p_entity in p_entitys}
        self.TP += len(t_entitys_set & p_entitys_set)

    #计算precision、recall、F1
    def em_count(self):
        
        tp = self.TP
        fp = self.P_sort - tp
        fn = self.T_sort - tp
        self.precision = tp / (tp + fp) if (tp + fp) else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) else 0.0
        p, r = self.precision, self.recall
        self.F1 = 2 * p * r / (p + r) if (p + r) else 0.0
    
    def get_Micro_data(self):
        self.em_count()
        return self.precision, self.recall, self.F1

if __name__ == "__main__":
    # em = EvaluationMetrics()
    pass