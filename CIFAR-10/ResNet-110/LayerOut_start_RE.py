from __future__ import print_function,absolute_import

import argparse
import os
import shutil
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import transforms
import torchvision.datasets as datasets
import argparse

from Arguments import get_args


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.init_weights()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def init_weights(self):
        if isinstance(self, nn.Conv2d):
            init.kaiming_normal(self.weight.data,mode='fanout')
            init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.Linear):
            init.kaiming_normal(self.weight.data,mode='fanout')
            init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.BatchNorm2d):
            init.kaiming_normal(self.weight.data,mode='fanout')
            init.constant_(self.bias.data, 0)
   




global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally

#args = parser.parse_args()
args = get_args()
state = {k: v for k, v in args._get_kwargs()}

device = torch.device("cuda" if args.cuda else "cpu")

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

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

def adjust_learning_rate(optimizer, epoch, count):
    global state
    lr=[0.01,0.1,0.01,0.001]
    if epoch in args.schedule:
    #if epoch != 0 and epoch % 50 == 0:
        state['lr'] = lr[count]
        #state['lr'] *= args.gamma 
        count=count+1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    return count

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_res110_freeze_start.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

best_acc1 = 0  # best test1 accuracy
best_epoch = 0
best_acc5 = 0

global best_acc1
global best_epoch
global best_acc5

start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint)



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print("==> creating model ")
   

model = ResNet(110) #for resnet20
    
model = model.to(device)
    
print(' \nTotal number of parameters: %.2f Million' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


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

# Resume

title = 'CIFAR10'
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



#assert False
model.train()

torch.save(model.state_dict(), 'res110_freeze_start_epochs')


del model
net = ResNet(110).to(device)

# load the weight
net.load_state_dict(torch.load('res110_freeze_start_epochs'))

K = 167
torch.manual_seed(23)
#global prob
import numpy as np

prob = torch.Tensor(K).uniform_(0,1)

print("probability : {}".format(prob))

def Freeze_train(trainloader, model, criterion, optimizer, epoch, use_cuda,prob,lr):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    
    # Making sure not all layers freezes
    while True:
        #survival = torch.bernoulli(prob)
        survival = torch.bernoulli(prob)
        if sum(list(filter(lambda p: p, survival))) == 0:
            continue
        else:
            break
    #print("probability : {}".format(prob))
    print("survival : {}".format(survival))


    #freezing weights according to survival prob
    index = 0
    for child in enumerate(net.children()):
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
    for child in net.modules():
        if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
            if (int(survival[index]) == 0):
                for param in child.parameters():
                    param.requires_grad = True
                index+=1

        

    return (losses.avg, top1.avg, prob)

# Train and val
count = 0
f_t_time = []
for epoch in range(start_epoch,args.epochs):
    count = adjust_learning_rate(optimizer, epoch,count)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    
    t = -time.time()
    train_loss, train_acc,prob = Freeze_train(trainloader, net, criterion, optimizer, epoch, use_cuda,prob,state['lr'])
    t += time.time()
    f_t_time.append(t)

    test_loss, test_acc,test_acc5 = test(testloader, net, criterion, epoch, use_cuda)

    print("Train Loss: ", train_loss)
    print("Test Loss: ", test_loss) 

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
