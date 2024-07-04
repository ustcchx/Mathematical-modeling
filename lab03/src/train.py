import processing_data
from models import MLPs
import torch.optim as opt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os
def train_func(lr: str, batch_size: int, num_epochs: int, activate: str, layer_num: int, layer_width: int):
    
    # 赋给初值
    # lr = 0.0003
    # batch_size = 16
    # size_list = [2, 1024, 1024, 1024, 1024, 3]
    # num_epochs = 1000
    # activate = "ReLU"
    
    lr = float(lr)
    size_list = [2] + [layer_width]*layer_num + [3]
    train_iter = processing_data.get_data_loader("train", batch_size, True)
    test_iter = processing_data.get_data_loader("test", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MLPs(size_list, activate)
    
    model.to(device)
    optimizer = opt.Adam(model.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    #criterion = nn.MSELoss()   # 定义损失函数MSE
    #criterion = nn.L1Loss()
    #criterion = nn.HuberLoss()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    counter = 0
    filename = "../checkpoints/checkpoint-0.pth"
    if os.path.exists(filename):
        while os.path.exists(f"../checkpoints/checkpoint-{counter}.pth"):
            counter += 1 
        filename = f"../checkpoints/checkpoint-{counter}.pth"
        
    log_file = f"../log/log-{counter}.txt"
    
    x, y = [], []
    max_acc_on_test =  0.0
    pbar = tqdm(total=num_epochs, desc="Processing", ncols=100)
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        for i,(inputs,true_y) in enumerate(train_iter):
            inputs = inputs.to(device)
            true_y = true_y.to(device) # 将输入和标签移至设备
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播           
            loss = criterion(outputs, true_y.type(torch.long).to(device)) # 计算损失
            loss.backward()  # 反向传播:
            optimizer.step()  # 更新模型参数
        x.append(epoch)
        y.append(float(loss))
        #scheduler.step()
        model.eval()
        running_loss = 0.0
        acc = 0
        for _,(inputs,true_y) in enumerate(test_iter):
            inputs = inputs.to(device)
            true_y = true_y.to(device) # 将输入和标签移至设备
            outputs = model(inputs)
            running_loss += criterion(outputs, true_y.type(torch.long).to(device))
            for k in range(len(outputs)) :
                s = torch.argmax(outputs[k])
                if s == true_y[k] :
                    acc +=1
        acc /= (test_iter.__len__()*batch_size)
        if acc > max_acc_on_test:
            choosen_epoch = epoch
            max_acc_on_test = acc
            checkpoint = {  
                'learning_rate': lr,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),   
                'loss': running_loss,  
            }
            torch.save(checkpoint, filename)
        with open(log_file, 'a') as f:  
            f.write("running loss on train datasets: " + str(float(loss)) + "\n")
            f.write(f"epoch {epoch}: acc: on test datasets: {acc}\n\n")
        pbar.update(1)
    
    pbar.close()
    plt.ylim(0, max(y)) 
    plt.plot(x, y, color="red")
    plt.savefig(f"../running_loss_pic/runing_loss_pic{counter}.png")
    return [ max_acc_on_test, choosen_epoch, f"../checkpoints/checkpoint-{counter}.pth", f"../running_loss_pic/runing_loss_pic{counter}.png"]
    
