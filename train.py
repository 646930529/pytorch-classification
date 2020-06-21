import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet_vision import resnet50 as resnet


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


EPOCH = 128
pre_epoch = 0
BATCH_SIZE = 32
LR = 0.001


net = resnet().to(device)
#net.load_state_dict(torch.load('./model/net_015.pth'))

transform_train = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainset = torchvision.datasets.ImageFolder(root='./data', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


classes = trainset.classes
print(trainset.classes)

print(len(trainset))
print(trainset[0][0].shape)
print(trainset[0][0].max())
print(trainset[0][0].min())

#vindex = 0
#for img in trainset:
#    vindex += 1
#    print(vindex, img[1], img[0].shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)



with open("log.txt", "a+")as f2:
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            f2.write('\n')
            f2.flush()

        print('Saving model......')
        torch.save(net.state_dict(), './model/net_%03d.pth' % (epoch + 1))

