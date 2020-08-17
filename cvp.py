import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
import glob
import cv2
import time
import os
import shutil


def removepath(path):
    time.sleep(0.1)
    if os.path.exists(path):
        shutil.rmtree(path)
    time.sleep(0.1)
    os.mkdir(path)
    time.sleep(0.1)


removepath('test1')
removepath('test2')
removepath('test3')
removepath('test4')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = models.resnet152(num_classes=3)
net.load_state_dict(torch.load('./model/net_100.pth'))
net.to(device)
net.eval()


transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


with torch.no_grad():
    findex = 0

    #cap = cv2.VideoCapture(0)
    #while 1:
    #    _, frame = cap.read()


    for imgpath in glob.glob('test/*'):
        frame = cv2.imread(imgpath)
        imgname = imgpath.replace('\\', '/').split('/')[-1]
        
        findex+=1
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        imgT = transform_test(img).unsqueeze(0).to(device)
        p = net(imgT).cpu().numpy()
        v = np.argmax(p)
        print(findex, v)

        cv2.imwrite('test'+str(v+1)+'/'+imgname, frame)
