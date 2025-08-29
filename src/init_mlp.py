# coding=gb2312
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    多层感知机生成器：将随机噪声映射为动力学参数。
    输入：噪声向量；输出：与 PARAMETER_NAMES 对应的参数值数组。
    """

    def __init__(self, input_size, output_size, hidden_sizes=[50, 50], device=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        layers = []
        # 输入层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Tanh())  # 使用Tanh激活函数

        # 隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Tanh())

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers).to(self.device)

    def get_weights(self):
        """
        返回当前网络参数的扁平化向量，依次包含所有权重和偏置。
        """
        # 从PyTorch模型中提取参数并展平
        flat_params = []
        for param in self.model.parameters():
            flat_params.append(param.data.view(-1))
        return torch.cat(flat_params).cpu().numpy()  # 确保返回numpy数组

    def set_weights(self, flat_weights):
        """
        将扁平化的权重向量赋值回各层权重和偏置。
        """
        # 将扁平化的numpy权重转换为torch tensor并分配给模型参数
        flat_weights_tensor = torch.from_numpy(flat_weights).float().to(self.device)

        start = 0
        for param in self.model.parameters():
            size = param.numel()
            param.data.copy_(flat_weights_tensor[start:start + size].view(param.size()))
            start += size

    def predict(self, X):
        """
        前向传播：计算网络输出。
        X 的形状为 (batch_size, input_size)。
        隐藏层使用双曲正切激活，输出层线性输出。
        """
        # 确保输入是torch tensor并移动到正确设备
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():  # 预测时不需要计算梯度
            output = self.model(X_tensor)
        return output.cpu().numpy()  # 确保返回numpy数组
