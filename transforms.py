import torchvision
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "transforms.jpeg"
img = Image.open(path)


transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.5, hue=[-0.25, 0.25]),
    transforms.ToTensor(),
])


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

images = []
for i in range(100):
    new_img = transform(img)
    print(new_img.shape)
    images.append(new_img)

images = torch.stack(images, 0)
print(images.shape)
imshow(torchvision.utils.make_grid(images, nrow = 10))

#while cv2.waitKey(1) != 27:
#    new_img = transform(img)
#    cv2_img = cv2.cvtColor(np.asarray(new_img),cv2.COLOR_RGB2BGR)
#    cv2.imshow(' ', cv2_img)

