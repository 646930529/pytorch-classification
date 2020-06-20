
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet18
import torch.onnx
from torch.autograd import Variable

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import glob



net = cv2.dnn.readNet('torch.onnx')
print(net)
dummy_input = np.zeros((1, 3, 32, 32))
net.setInput(dummy_input)
preds = net.forward()
print(preds)



ccc = {0:0, 1:0}
ci = 0
for img in glob.glob('test/dog/*.jpg'):

    img = cv2.imread(img)
    npdata = img[:,:,::-1]
    
    blob = npdata.astype(np.float32) / 255
    blob = cv2.dnn.blobFromImage(blob, 1, (32,32), (0.5, 0.5, 0.5)) / 0.5

    net.setInput(blob)
    preds = net.forward()
    v = np.argmax(preds)
    ccc[v] += 1
    print(ci)
    ci += 1
print(ccc)


