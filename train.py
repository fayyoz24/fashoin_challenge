from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from model import Classify
from torch import optim
import matplotlib.pyplot as plt
from helper_functions import acc
import data_handler as dh



# Download and load the training data
dh.trainset
dh.trainloader

# Download and load the test dataclear

dh.testset
dh.testloader


clf = Classify(784,10)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(clf.parameters(),lr=0.05, momentum=0.9)

epochs = 21
losses = []
test_losses = []


for e in range(epochs):
    running_loss=0

    for images, labels in iter(dh.trainloader):
        
        #images.resize(images.size()[0],784)
        images=images.view(images.shape[0],784)
        
        optimizer.zero_grad()
        pred=clf.forward(images)
        loss=criterion(pred,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f'Number of Epochs {e}/{epochs-1}')
    print('Train Loss:  ', running_loss/len(dh.trainloader))
    losses.append(running_loss/len(dh.trainloader))
    clf.eval()
    with torch.no_grad():
        test_running_loss = 0
        for images_test, labels_test in iter(dh.trainloader):

            images_test = images_test.view(images_test.shape[0], -1)
            test_pred = clf.forward(images_test)
            test_loss = criterion(test_pred, labels_test)
            test_running_loss += test_loss.item()

        print('Test Loss:   ', test_running_loss/len(dh.testloader))
        test_losses.append(test_running_loss/len(dh.testloader))
        
        #if e % 5 == 0:
        print('Accuracy', acc(test_pred,labels_test))
            
    clf.train()

    torch.save(clf.state_dict(), 'classifer.pth')


plt.plot(losses, label='trainlosses')
plt.plot(test_losses, label='testlosses')   
plt.legend()
plt.savefig('losses.png')
plt.show()