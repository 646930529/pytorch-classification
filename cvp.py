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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = models.resnet50(num_classes=3)
net.load_state_dict(torch.load('./net_099.pth'))
net.to(device)
net.eval()


transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

import cv2
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        #img = Image.open(file)
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        imgT = transform_test(img).unsqueeze(0).to(device)
        p = net(imgT).cpu().numpy()
        v = np.argmax(p)
        print(['photo','phone','normal'][v])
