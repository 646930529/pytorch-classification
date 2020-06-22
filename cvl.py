
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

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
dummy_input = np.zeros((1, 3, 224, 224))
net.setInput(dummy_input)
preds = net.forward()
print(preds)



def testfile():
    ccc = {0:0, 1:0}
    ci = 0
    for img in glob.glob('data/cup/*.jpg'):
        img = cv2.imread(img)
        if img is None:
            continue
        cv2.imshow('1',img)
        cv2.waitKey(1)

        npdata = img[:,:,::-1]
        
        blob = npdata.astype(np.float32) / 255
        blob = cv2.dnn.blobFromImage(blob, 1, (224, 224), (0.5, 0.5, 0.5)) / 0.5

        net.setInput(blob)
        preds = net.forward()
        v = np.argmax(preds)
        print(v,preds)
        ccc[v] += 1
        print(ci)
        ci += 1
        if ci > 100:
            break
    print(ccc)


def testcap():
    cap = cv2.VideoCapture(0)
    while 1:
        _, img = cap.read()
        if img is None:
            continue
        img = img[100:300,100:300]
        cv2.imshow('1',img)
        cv2.waitKey(1)

        npdata = img[:,:,::-1]
        
        blob = npdata.astype(np.float32) / 255
        blob = cv2.dnn.blobFromImage(blob, 1, (224, 224), (0.5, 0.5, 0.5)) / 0.5

        net.setInput(blob)
        preds = net.forward()
        v = np.argmax(preds)
        print(v,preds)
        
        img = blob[0].transpose(1,2,0)
        img = img / 2 + 0.5
        img = img[:,:,::-1]
        cv2.imshow('2',img)
        cv2.waitKey(1)
        
testcap()

