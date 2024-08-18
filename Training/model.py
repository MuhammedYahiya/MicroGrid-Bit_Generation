import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity

def create_model(input_size, hidden1, hidden2, weight_bit_width, act_bit_width, num_classes):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.BatchNorm1d(input_size),
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden1),
        nn.Dropout(0.3),
        QuantReLU(bit_width=act_bit_width),

        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden2),
        nn.Dropout(0.3),
        QuantReLU(bit_width=act_bit_width),

        QuantLinear(hidden2, num_classes, bias=True, weight_bit_width=weight_bit_width)
    )
    return model

