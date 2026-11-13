import json

def view(path):
    line_idx = 0
    False_st = []
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        line_idx += 1
        #出错标志
        error_flag = False
        #实体开始标志
        entity_flag = False
        entity_end = ""

        while line:
            context = line.strip().split()
            if not context:
                line = f.readline()
                line_idx += 1
                continue
            BIO_context = context[1]

            #出错之后要找到第一个"O"
            if error_flag:
                if BIO_context == "O":
                    error_flag = False
            else:
                #实体开始
                if BIO_context.startswith("B"):
                    entity_end = BIO_context.split('-')[1]
                    entity_flag = True
                elif BIO_context.startswith("I"):
                    if entity_flag:
                        if BIO_context.split('-')[1] != entity_end:
                            False_st.append(line_idx)
                            error_flag = True
                            entity_flag = False
                    else:
                        False_st.append(line_idx)
                        error_flag = True
                        entity_flag = False
                elif BIO_context == "O":
                    if entity_flag:
                        entity_flag = False
                

            line = f.readline()
            line_idx += 1
    return False_st

def modify(path, index):
    with open(path, 'r', encoding='utf-8') as f:
        # 读取全部内容
        text = f.readlines()

    for i, line in enumerate(text, start = 1):
        if i in index:
            text[i - 1] = line.replace('I-', 'B-', 1)
    #写回修改后的内容
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(text)

def check(config):
    dataset_name = {'train', 'dev', 'test'}
    option = int(input("仅查看出错内容输入(1), 修改出错内容输入(2):"))

    for name, paths in config.items():
        print(f"---------------------------------正在检查{name}数据集---------------------------------")
        for d_class, path in paths.items():
            if d_class in dataset_name:
                print(f"正在检查{d_class}文件")
                false_st = view(path)
                
                if false_st:
                    print(f"出错内容下标:{false_st}")
                    if option == 2:
                        modify(path, set(false_st))
                        print(f"已修改{d_class}的错误BIO标签!!!")
                else:
                    print(f"无出错内容下标")


if __name__ == "__main__":
    options_path = './config/options.json'
    with open(options_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    check(config["datasets"])