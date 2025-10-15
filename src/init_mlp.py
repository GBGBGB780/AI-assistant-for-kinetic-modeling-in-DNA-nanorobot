# coding=gb2312
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    ����֪�������������������ӳ��Ϊ����ѧ������
    ���룺����������������� PARAMETER_NAMES ��Ӧ�Ĳ���ֵ���顣
    """

    def __init__(self, input_size, output_size, hidden_sizes=[50, 50], device=None,
                 param_names=None, param_ranges=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        default_min, default_max = param_ranges['default_range']
        min_vals = []
        max_vals = []
        for name in param_names:
            min_val, max_val = param_ranges.get(name, (default_min, default_max))
            min_vals.append(min_val)
            max_vals.append(max_val)

        # ����Χת��Ϊ PyTorch �������洢���Ա��� predict �и�Чʹ��
        self.min_vals = torch.tensor(min_vals, dtype=torch.float32, device=self.device)
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32, device=self.device)

        layers = []
        # �����
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Tanh())  # ʹ��Tanh�����

        # ���ز�
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Tanh())

        # �����
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers).to(self.device)

    def get_weights(self):
        """
        ���ص�ǰ��������ı�ƽ�����������ΰ�������Ȩ�غ�ƫ�á�
        """
        # ��PyTorchģ������ȡ������չƽ
        flat_params = []
        for param in self.model.parameters():
            flat_params.append(param.data.view(-1))
        return torch.cat(flat_params).cpu().numpy()  # ȷ������numpy����

    def set_weights(self, flat_weights):
        """
        ����ƽ����Ȩ��������ֵ�ظ���Ȩ�غ�ƫ�á�
        """
        # ����ƽ����numpyȨ��ת��Ϊtorch tensor�������ģ�Ͳ���
        flat_weights_tensor = torch.from_numpy(flat_weights).float().to(self.device)

        start = 0
        for param in self.model.parameters():
            size = param.numel()
            param.data.copy_(flat_weights_tensor[start:start + size].view(param.size()))
            start += size

    def predict(self, X):
        """
        ǰ�򴫲����������������
        X ����״Ϊ (batch_size, input_size)��
        ���ز�ʹ��˫�����м����������������
        """
        # ȷ��������torch tensor���ƶ�����ȷ�豸
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():  # Ԥ��ʱ����Ҫ�����ݶ�
            output = self.model(X_tensor)
        # ���з�Χ����
        scaled_output = (output + 1) / 2 * (self.max_vals - self.min_vals) + self.min_vals

        return scaled_output.cpu().numpy()
