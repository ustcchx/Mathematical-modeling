# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 02:01:48 2024

@author: 28577
"""
import gradio as gr
from img_process import image_repair

with gr.Interface(fn=image_repair, 
                  inputs=["image",
                          gr.Slider(value=2000, minimum=0, maximum=5000, step=100),
                          gr.Slider(value=200, minimum=0, maximum=500, step=10),
                          "text"
                          ], 
                  outputs=["image", "image"], 
                  examples=[  
                       [  # 这是一个列表，其中包含了所有输入的示例值  
                          "../figures/test-7.png",  # 图片路径  
                          800,  # in_max_iter值  
                          200,   # out_max_iter值  
                          "0.001"  # tol的文本值  
                      ]  
                  ]) as demo:  
 demo.launch()