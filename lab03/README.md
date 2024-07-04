## 实验三代码使用步骤
### 0. python相关包以及版本需求
~~~
torch GPU版本
tqdm 
gradio == 3.41.2
# 这里限制gradio版本是因为代码中加入了与该版本前端容器匹配的外部CSS样式
~~~

### 1. 命令行进入src目录后，启动程序gradio UI
~~~
python run_web.py
~~~
得到如下类似URL：
~~~
Running on local URL:  http://127.0.0.1:7862
~~~
在浏览器中打开即可

### 2. 调整网络超参数
可以调整batch size, learning rate, max epoch等参数，提交后等待训练与测试结果。建议使用默认参数（经过笔者的调参所得），4070显卡配置大概2-3min训练测试完成。

### 3. 得到训练结果
训练测试结束后，在log文件夹中存有训练过程中的输出，checkpoints文件夹中存有当此训练表现最佳的模型内部参数，running_loss_pic中存有训练损失变化图像。如果希望再次考察已训练模型，需要使用test.py，同时需要根据checkpoint文件路径、以及对应的模型，来更改pth路径、模型结构和激活函数，调整后直接运行即可。