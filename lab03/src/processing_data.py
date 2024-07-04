import torch.utils.data as data
import torch
from torch import tensor 

class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index] 
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)

def get_data_loader(file_type: str, batch_size: int, shuffle=False):
    x_tensor_list = []
    y_tensor_list = []
    if file_type == "train":
        file_1 = "insects-training.txt"
        file_2 = "insects-2-training.txt"
    elif file_type == "test":
        file_1 = "insects-testing.txt"
        file_2 = "insects-2-testing.txt"
    with open(f"../insects/{file_1}", 'r') as file:
    # 逐行读取
        for line in file:
        # 假设每行数据以空格分隔
            data_ = line.strip().split()
            x_tensor_list.append(tensor([float(data_[0]), float(data_[1])], dtype=torch.float))
            y_tensor_list.append(tensor(int(data_[2]), dtype=torch.float))
    with open(f"../insects/{file_2}", 'r') as file:
    # 逐行读取
        for line in file:
        # 假设每行数据以空格分隔
            data_ = line.strip().split()
            x_tensor_list.append(tensor([float(data_[0]), float(data_[1])], dtype=torch.float))
            y_tensor_list.append(tensor(int(data_[2]), dtype=torch.float))
    dataset = Mydataset(x_tensor_list, y_tensor_list)
    return data.DataLoader(dataset, batch_size=batch_size)


