import torch
import torchvision
import torchvision.transforms as transforms
# imports
import matplotlib.pyplot as plt
import numpy as np


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# transforms
transform = transforms.Compose(
    [
    #transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

print(trainset.classes)
print(len(trainset))
print(trainset[0])
#print(trainset[0][0].size())
#print(trainset[0][0].max())
#print(trainset[0][0].min())

index = 0
for i in trainset:

    if i[1] == 3:
        ...
        print('cat')
        i[0].save('data2/cat/'+str(index)+'.jpg')
    if i[1] == 5:
        ...
        print('dog')
        
        i[0].save('data2/dog/'+str(index)+'.jpg')

    print(index)
    index += 1


