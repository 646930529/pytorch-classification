
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from resnet_vision import resnet50 as resnet
import torch.onnx
from torch.autograd import Variable

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

net = resnet().to(device)
#net.load_state_dict(torch.load('./model/net_013.pth'))
net.eval()

with torch.no_grad():
    dummy_input = Variable(torch.zeros(1, 3, 112, 112))
    print(net(dummy_input).numpy())
    torch.onnx.export(net, dummy_input, "torch.onnx")

