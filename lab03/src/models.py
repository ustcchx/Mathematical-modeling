import torch.nn as nn

# 前馈神经网络模型
class MLPs(nn.Module):
    
    def __init__(self, size_list: list, activate: str):
        super(MLPs, self).__init__()
        self.size_list = size_list
        self.activate = activate
        self.MLPs_layer = self.__make_layer()
    
    def __make_layer(self):
        layers = []
        for i in range(len(self.size_list))[:-1]:
            layers.append(nn.Linear(self.size_list[i], self.size_list[i + 1]))
            if i != len(self.size_list)-2:
                if self.activate == "ReLU":
                    layers.append(nn.ReLU())
                elif self.activate == "LeakyReLU":
                    layers.append(nn.LeakyReLU())
                elif self.activate == "Tanh":
                    layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.MLPs_layer(x)
        return x