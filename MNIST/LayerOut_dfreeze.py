
# coding: utf-8

# In[1]:


from __future__ import print_function,absolute_import

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

import numpy as np

from Arguments import get_args


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32, 64, 3,stride=2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,64,3, stride=2)
        self.conv5 = nn.Conv2d(64,32,3)
        self.fc1 = nn.Linear(32 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.initWeights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def initWeights(self):
        if isinstance(self, nn.Conv2d):
            init.xavier_normal_(self.weight.data)
            init.normal_(self.bias.data)
        elif isinstance(self, nn.Linear):
            init.xavier_normal_(self.weight.data)
            init.normal_(self.bias.data)
        elif isinstance(self, nn.BatchNorm2d):
            init.normal_(self.weight.data, mean=1, std=0.02)
            init.constant_(self.bias.data, 0)


# In[3]:


global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally

#args = parser.parse_args()
args = get_args()
state = {k: v for k, v in args._get_kwargs()}

device = torch.device("cuda" if args.cuda else "cpu")

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


# In[4]:


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[5]:


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[6]:


def adjust_learning_rate(optimizer, epoch, count):
    global state
    #lr=[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    if epoch in args.schedule:
        #state['lr'] = lr[count]
        state['lr'] *= args.gamma 
        count=count+1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    return count


# In[7]:


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_H2L.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# In[8]:


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

best_acc1 = 0  # best test1 accuracy
best_epoch = 0
best_acc5 = 0


# In[9]:


global best_acc1
global best_epoch
global best_acc5

start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint)


# In[10]:


# Data Augmentation

print('==> Preparing dataset %s' % args.dataset)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

dataloader = datasets.MNIST

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


# In[11]:


# Model
print("==> creating model ")
   

model = Net()

model = model.to(device)

print(' \nTotal number of parameters: %.2f Million' % (sum(p.numel() for p in model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


# In[12]:


# Resume

title = 'MNIST'
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc1 = checkpoint['best_acc1']
    best_acc5 = checkpoint['best_acc5']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
elif args.evaluate:
    print('\nEvaluation only')
    test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))


# In[13]:


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    print("train")
     
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
                
        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs = inputs.to(device)
        
        targets = targets.to(device)
               
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx%100==0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s |Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
        
    return (losses.avg, top1.avg,top5.avg)
    


# In[14]:


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    print("test")
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
       
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print('Test_Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
    return (losses.avg, top1.avg, top5.avg)


# In[15]:


# Train and val
count = 0
for epoch in range(start_epoch, start_epoch+20):
    count = adjust_learning_rate(optimizer, epoch,count)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    
    train_loss, train_acc1, train_acc5 = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc1, test_acc5 = test(testloader, model, criterion, epoch, use_cuda)

    # save model
    is_best = test_acc1 > best_acc1
    best_acc1 = max(test_acc1, best_acc1)
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc1,
                'best_acc': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    print('Best acc1:')
    print(best_acc1)
    


# In[16]:


model.train()
torch.save(model.state_dict(), 'FreezeOut_12layer_H2L_20epochs')


# In[17]:


#del model
net = Net().to(device)

# load the weight
net.load_state_dict(torch.load('FreezeOut_12layer_H2L_20epochs'))


# In[18]:


K = 7 # no.of layers 
torch.manual_seed(23)
#global prob
#prob = torch.Tensor(K).uniform_(0,1)
prob = torch.from_numpy(np.array([0.9,0.7,0.6,0.5,0.4,0.3,0.1]))



# In[19]:


def Freeze_train(trainloader, model, criterion, optimizer, epoch, use_cuda,prob,lr):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    
    # Making sure not all layers freezes
    while True:
        survival = torch.bernoulli(prob)
        if sum(list(filter(lambda p: p, survival))) == 0:
            continue
        else:
            break
    print("probability : {}".format(prob))
    print("survival : {}".format(survival))


    #freezing weights according to survival prob
    index = 0
    for child in net.children():
        if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
            if (int(survival[index]) == 0):
                for param in child.parameters():
                    param.requires_grad = False
                index+=1

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=lr,momentum = 0.9, weight_decay=5e-4)

    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    print("train")
     
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
                
        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
               
        # compute output
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx%100==0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s |Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
        
    
    # Unfreezing the frozen layers
    index = 0
    for child in net.children():
        if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
            if (int(survival[index]) == 0):
                for param in child.parameters():
                    param.requires_grad = True
                index+=1

        

    return (losses.avg, top1.avg, prob)


# In[20]:


# Train and val
count = 1
for epoch in range(start_epoch+20, 100):
    count = adjust_learning_rate(optimizer, epoch,count)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    train_loss, train_acc,prob = Freeze_train(trainloader, net, criterion, optimizer, epoch, use_cuda,prob,state['lr'])
    test_loss, test_acc,test_acc5 = test(testloader, net, criterion, epoch, use_cuda)

    # save model
    is_best = test_acc > best_acc1
    best_acc1 = max(test_acc, best_acc1)
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc1)



