import torch
from torch import nn
from typing import Callable
from . import surrogate

class LIFNode(nn.Module):
    def __init__(self, tau: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan()):
        """
        :param tau: 神经元膜电位衰减率，值越大衰减越慢
        :type tau: float
        """
        # TODO assert判断一下参数，不能为None之类的
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function

        # self.v代表该层神经元的膜电位，初始时用一个v_reset标量表示初始值，
        # 用前一层为Linear(input, output)为例，之后的改变量都是一个batch_size*output的矩阵，
        # 利用torch的广播机制进行运算，改变之后self.v的形状也是shape(batch_size, output)
        self.v = v_reset


    def neuronal_charge(self, x: torch.Tensor):
        """
        :param x: 输入瞬时电流矩阵，shape(batch_size, output)
        :type x: torch.Tensor

        神经元模型充电函数
        """
        self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def neuronal_fire(self):
        """
        根据当前神经元的电压、阈值，计算输出脉冲。
        """
        return self.surrogate_function(self.v - self.v_threshold)
    
    def neuronal_reset(self, spike: torch.Tensor):
        """
        根据本层神经元当前时刻的发放脉冲，reset掉发放脉冲的神经元
        spike: shape(batch_size, output)
        """
        self.v = (1. - spike) * self.v + spike * self.v_reset

    def step_forward(self, x: torch.Tensor):
        """
        输入电流，前向传播进行运算
        :param x: 输入到神经元的膜电位增量
        :type x: torch.Tensor
        :return: 神经元层的输出脉冲
        :rtype: torch.Tensor
        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def forward(self, x: torch.Tensor):
        return self.step_forward(x)
    
    def reset(self):
        self.v = self.v_reset