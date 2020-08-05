
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
import time
import shutil
import os


net = cv2.dnn.readNet('torch.onnx')
print(net)
dummy_input = np.zeros((1, 3, 224, 224))
net.setInput(dummy_input)
preds = net.forward()
print(preds)


def removepath(path):
    time.sleep(0.1)
    if os.path.exists(path):
        shutil.rmtree(path)
    time.sleep(0.1)
    os.mkdir(path)
    time.sleep(0.1)


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    if cv_img is None:
        return None
    #print('cv_imread', cv_img.shape)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGRA2RGB)
    return cv_img


def testfile():
    removepath('test1')
    removepath('test2')
    removepath('test3')
    removepath('test4')

    ccc = {0:0, 1:0, 2:0, 3:0}
    ARR = []
    ci = 0
    for file in glob.glob('test/*'):
        #print(ci,img)
        img = cv_imread(file)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        #cv2.imshow('1', img)
        cv2.waitKey(1)

        npdata = img[:,:,::-1]
        #print(npdata.shape, npdata.mean(), npdata.max(), npdata.min())
        
        blob = npdata.astype(np.float32) / 255
        blob = cv2.dnn.blobFromImage(blob, 1, (224, 224), (0.5, 0.5, 0.5)) / 0.5

        #print(blob.shape)
        net.setInput(blob)
        preds = net.forward()
        v = np.argmax(preds)
        if preds.max() < 0.1:
            v = 3
        print(file, ci,v,preds)
        ARR.append(preds[0])
        ccc[v] += 1
        ci += 1
        file = file.replace('\\','/').split('/')[-1]
        cv2.imwrite('test'+str(v+1)+'/'+file+'_'+str(time.time())+'.jpg', img)
        
        img = blob[0].transpose(1,2,0)
        img = img / 2 + 0.5
        img = img[:,:,::-1]
        #cv2.imshow('2',img)
        cv2.waitKey(1)
    print(ccc)
    np.save('ARR.npy', ARR)


def testcap():
    cap = cv2.VideoCapture(0)
    while 1:
        _, img = cap.read()
        if img is None:
            continue
        cv2.imshow('1',img)
        cv2.waitKey(1)
        img = img[150:374,150:374]

        npdata = img[:,:,::-1]
        
        blob = npdata.astype(np.float32) / 255
        blob = cv2.dnn.blobFromImage(blob, 1, (224, 224), (0.5, 0.5, 0.5)) / 0.5

        net.setInput(blob)
        preds = net.forward()
        v = np.argmax(preds)
        if preds.max() < 0.1:
            v = 3
        print(v,preds)
        
        img = blob[0].transpose(1,2,0)
        img = img / 2 + 0.5
        img = img[:,:,::-1]
        cv2.imshow('2',img)
        cv2.waitKey(1)

testfile()
#testcap()

