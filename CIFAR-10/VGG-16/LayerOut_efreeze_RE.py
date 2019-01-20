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
import transforms
import torchvision.datasets as datasets
import argparse

from Arguments import get_args



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.initWeights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc3(self.fc2(self.fc1(out)))
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def initWeights(self):
        if isinstance(self, nn.Conv2d):
            init.kaiming_normal(self.weight.data,mode='fanout')
            init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.Linear):
            init.normal_(self.weight.data,std=1e-3)
            init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.BatchNorm2d):
            init.normal_(self.weight.data, mean=1, std=0.02)
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
    #lr=[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    if epoch in args.schedule:
    #if epoch != 0 and epoch % 50 == 0:
        #state['lr'] = lr[count]
        state['lr'] *= args.gamma 
        count=count+1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    return count


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_vgg16_new_freezeL2H_RE.pth.tar'):
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print("==> creating model ")
   

model = VGG("VGG16")
    
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


#Train and val
count = 0
train_time = []
for epoch in range(start_epoch, start_epoch+20):
    count = adjust_learning_rate(optimizer, epoch,count)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    t = -time.time()
    train_loss, train_acc1, train_acc5 = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
    t += time.time()
    train_time.append(t)
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
    
model.train()
torch.save(model.state_dict(), 'FreezeOut_vgg_20_L2H_REepochs')



del model
net = VGG("VGG16").to(device)

# load the weight
net.load_state_dict(torch.load('FreezeOut_vgg_20_L2H_REepochs'))


K = 16 # no.of layers 
torch.manual_seed(23)
#global prob
import numpy as np

prob = torch.from_numpy(np.array([0.1,0.2,0.2,0.3,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9]))
f_counter = np.zeros(K)

def Freeze_train(trainloader, model, criterion, optimizer, epoch, use_cuda,prob,lr,f_counter):
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
    print("probability : {}".format(prob))
    print("survival : {}".format(survival))

    idx = np.where(survival==0)[0]
    for i in range(len(idx)):
        f_counter[idx[i]] +=1


    #freezing weights according to survival prob
    index = 0
    for child in net.modules():
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
count = 1
f_t_time = []
for epoch in range(start_epoch+20, 300):
    count = adjust_learning_rate(optimizer, epoch,count)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    
    t = -time.time()
    train_loss, train_acc,prob = Freeze_train(trainloader, net, criterion, optimizer, epoch, use_cuda,prob,state['lr'],f_counter)
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


print("______"*30)
print("LayerWise : {}".format(f_counter))
print("\n Total No. of parameters: {}".format(sum(f_counter)))




