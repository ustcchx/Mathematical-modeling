import processing_data
from models import MLPs
import torch

def main():
    
    batch_size = 16
    size_list = [2, 1024, 1024, 1024, 1024, 3]
    activate = "ReLU"
    checkpoint = torch.load("../checkpoints/checkpoint-0.pth")
    test_iter = processing_data.get_data_loader("test", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MLPs(size_list, activate)
    model.load_state_dict(checkpoint['model_state_dict']) 
    model.to(device)
    acc = 0
    for _,(inputs,true_y) in enumerate(test_iter):
        inputs = inputs.to(device)
        true_y = true_y.to(device) # 将输入和标签移至设备
        outputs = model(inputs)
        for k in range(len(outputs)) :
            s = torch.argmax(outputs[k])
            if s == true_y[k] :
                acc += 1
    acc /= (test_iter.__len__()*batch_size)
    print(f"accuracy on test datasets: {acc}")

if __name__ == '__main__':  
    main()