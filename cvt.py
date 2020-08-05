import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.autograd import Variable


#net = models.resnet152(num_classes=3)
#net = models.mobilenet_v2(num_classes=3)
net = models.resnext101_32x8d(num_classes=3)

net.load_state_dict(torch.load('./model/net_100.pth', map_location={'cuda:3':'cuda:0'}))
net.eval()


with torch.no_grad():
    dummy_input = Variable(torch.zeros(1, 3, 224, 224))
    print(net(dummy_input).numpy())
    torch.onnx.export(net, dummy_input, "torch.onnx")

