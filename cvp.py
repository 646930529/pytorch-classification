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
net = models.mobilenet_v2(num_classes=3)
net.load_state_dict(torch.load('./model/net_090.pth'))
net.to(device)
net.eval()


transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


with torch.no_grad():
    file = r'D:\fire_smoke\test\ta (22).jpg'
    img = Image.open(file)
    imgT = transform_test(img).unsqueeze(0).to(device)
    p = net(imgT).cpu().numpy()
    v = np.argmax(p)
    print(['fire','smoke','normal'][v])
