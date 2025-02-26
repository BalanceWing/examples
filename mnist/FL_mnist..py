###############################################################################
#
#  Distributed Parallel training framework 
#  Version 1.0 created by t05885
#  2019.6.18 
#
###############################################################################

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import time

import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
#def train(args, model, device, train_loader, optimizer, epoch):
def train(args, model, optimizer, train_loader, epoch):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = F.nll_loss(output, target)
        
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#def test(args, model, device, test_loader):
def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    #distributed parralled setting
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:9099', 
                        help='master host and port')
    parser.add_argument('--rank', type=int, 
                        help='training process number and priority')
    parser.add_argument('--world-size', type=int, 
                        help='schedule allover training processes')
                        
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    #initialized distributed 
    dist.init_process_group (init_method = args.init_method, backend = "gloo", world_size = args.world_size, rank = args.rank, group_name = "pytorch_dist0")

    torch.manual_seed(args.seed)
    if use_cuda :
        torch.cuda.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")
    
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    #distribute data
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True, 
        **kwargs)
        
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, 
        shuffle=True, 
        **kwargs)


    #model = Net().to(device)
    model = Net()
    if use_cuda:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel (model)
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    #train and metric
    total_time = 0

    for epoch in range(1, args.epochs + 1):
        #set epoch , for syn all status
        train_sampler.set_epoch (epoch)
        
        start_cpu_secs = time.time()
        #train(args, model, device, train_loader, optimizer, epoch)
        train(args, model, optimizer, train_loader, epoch)
        end_cpu_secs = time.time()
        
        print ("Epoch {} of {} took {:.3f}s".format(
            epoch, args.epochs, end_cpu_secs -start_cpu_secs))
        
        total_time += end_cpu_secs - start_cpu_secs
        
        #test(args, model, device, test_loader)
        test (args, model, test_loader)
        
        print ("Total time = {:.3f}s".format(total_time) )
    
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
    
