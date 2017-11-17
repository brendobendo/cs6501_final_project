import torch, random
import torchvision
from torchvision.datasets import CIFAR10 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn 
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import json, string
import numpy as np
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser(description='train vgg16 net on the desired dataset')
parser.add_argument('--pretrained', dest='pretrained', help='specfity to use pretrained vgg16 model', default=False, action='store_true')
parser.add_argument('--learningRate', help='input the learning rate', default=1e-2, type=float)
parser.add_argument('--batchSize', help='specify mini batch size', default=16, type=int)
parser.add_argument('--num_workers', help='specify number of workers in data loader', default=2, type=int)
parser.add_argument('--n_epochs', help='specify number of training epochs', default=10, type=int)
parser.add_argument('--gpu', help='use gpu',default=False, action='store_true')
parser.add_argument('--checkPoint', help='specify check point name', type=str)

args = parser.parse_args()
learningRate = args.learningRate
pre = args.pretrained
bz = args.batchSize
workers = args.num_workers
epochs = args.n_epochs
gpu = args.gpu
checkPoint = args.checkPoint


vgg16 = torchvision.models.vgg16(pretrained=pre)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr = learningRate)

imgTransform = transforms.Compose([transforms.Scale(224),
                                   transforms.RandomCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])

def getLoader(trainRoot, valRoot, imgTransform, bz, workers):
    CFTrainSet = torchvision.datasets.ImageFolder(root=trainRoot, transform=imgTransform)
    CFValSet = torchvision.datasets.ImageFolder(root=valRoot, transform=imgTransform)

    CFTrainLoader = torch.utils.data.DataLoader(CFTrainSet, batch_size=bz, shuffle=True, num_workers=workers)
    CFValLoader = torch.utils.data.DataLoader(CFValSet, batch_size=bz, shuffle=True, num_workers=workers)
    return CFTrainLoader, CFValLoader


# Train the previously defined model.
def train_model(network, criterion, optimizer, checkPoint, trainLoader, valLoader, n_epochs = 10, use_gpu = False):
    train_acc = list()
    val_acc = list()
    train_loss = list()
    val_loss = list()
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, labels)) in enumerate(t):
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
        
        train_loss.append(cum_loss / (1 + i))
        train_acc.append(100 * correct / counter)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
        }, checkPoint + '.pth.tar' )

        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
            
        val_loss.append(cum_loss / (1 + i))
        val_acc.append(100 * correct / counter)
        
        epochs = np.arange(0.0, n_epochs, 1.0)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(epochs, train_acc, '-o', label = 'train_acc' )
    plt.plot(epochs, val_acc, '-o', label = 'validate_acc')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.subplot(212)
    plt.plot(epochs, train_loss, '-o', label = 'train_loss')
    plt.plot(epochs, val_loss, '-o', label = 'validate_loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()


CFTrainLoader, CFValLoader = getLoader("./all_data_w_splits/train", "./all_data_w_splits/val", imgTransform, bz, workers)
train_model(vgg16, criterion, optimizer, checkPoint, CFTrainLoader, CFValLoader, n_epochs = epochs, use_gpu = gpu)






