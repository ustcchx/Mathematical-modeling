import gradio as gr
from train import train_func
with open('style.css', 'r', encoding="UTF-8") as file:  
    custom_css = file.read()
with gr.Interface(
        fn=train_func, 
        inputs=[
            gr.Textbox(value="0.0003", label="学习率"),
            gr.Slider(value=16, minimum=0, maximum=2048, step=1),
            gr.Slider(value=1000, minimum=0, maximum=10000, step=50),
            gr.Radio(["ReLU", "Tanh", "LeakyReLU"], label="激活函数", info="激活函数类型", value = "ReLU"),
            gr.Slider(value=4, minimum=1, maximum=20, step=1, label="网络隐藏层个数", info="网络隐藏层个数"),
            gr.Slider(value=1024, minimum=8, maximum=2024, step=2, label="隐藏层每层神经元个数", info="隐藏层每层神经元个数"),
        ], 
        outputs=[ gr.Textbox(label="在测试数据集上的准确率"),
                 gr.Textbox(label="在测试集上表现优异的模型对应的epoch"), 
                 gr.Textbox(label="本次训练中在训练集上表现最优异的模型参数存入以下路径"),
                 gr.Image(label="每一个epoch损失展示")],
        css=custom_css
    ) as demo:  
 demo.launch()