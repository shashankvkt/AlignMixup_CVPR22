import os
import random
import numpy as np
import argparse
import itertools
from models.resnet18_classifier import Resnet_classifier

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from utils.dataLoader import dataloader_train, dataloader_val


parser = argparse.ArgumentParser(description='Trains ResNet on tiny-imagenet-200', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_dir', type = str, default = '/nfs/pyrex/raid6/svenkata/Datasets/tiny-imagenet-200/train',
                        help='path to train folder')
parser.add_argument('--val_dir', type = str, default = '/nfs/pyrex/raid6/svenkata/Datasets/tiny-imagenet-200/val',
                        help='path to val folder')
parser.add_argument('--save_dir', type = str, default = '/nfs/pyrex/raid6/svenkata/weights/AlignMixup_CVPR22/tiny_imagenet/',
                        help='folder where results are to be stored')

# Optimization options

parser.add_argument('--epochs', type=int, default=1200, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=2.0, help='alpha parameter for mixup')
parser.add_argument('--num_classes', type=int, default=200, help='number of classes, set 100 for CIFAR-100')

parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr_', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[600, 900], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()

out_str = str(args)
print(out_str)



device = torch.device("cuda" if args.ngpu>0 and torch.cuda.is_available() else "cpu")

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)



transform_train = transforms.Compose([
	transforms.RandomCrop(64, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465),
	                     (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465),
	                     (0.2023, 0.1994, 0.2010)),
])



train_data = dataloader_train(args.train_dir, transform_train)
test_data = dataloader_val(os.path.join(args.val_dir, 'images'), os.path.join(args.val_dir, 'val_annotations.txt'), transform_test)

trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)


model = Resnet_classifier(num_classes=args.num_classes)
model = torch.nn.DataParallel(model)
model.to(device)
print(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr_, momentum=args.momentum, weight_decay=args.decay)
criterion = nn.CrossEntropyLoss()
best_acc = 0

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        print("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



def train(epoch):	
	model.train()
	total_loss = 0
	correct = 0
	for i, (images, targets) in enumerate(trainloader):
		
		images = images.to(device)
		targets = targets.to(device)
		
		lam = np.random.beta(args.alpha, args.alpha)
		
		outputs,targets_a,targets_b = model(images, targets, lam, mode='train')
		
		loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		total_loss += loss.item()

		_,pred = torch.max(outputs, dim=1)
		correct += (pred == targets).sum().item()
	
	print('epoch: {} --> Train loss = {:.4f} Train Accuracy = {:.4f} '.format(epoch, total_loss / len(trainloader.dataset), 100.*correct / len(trainloader.dataset)))
	


def test(epoch):
	#best_acc = 0
	global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			
			inputs = inputs.to(device)
			targets = targets.to(device)
			
			outputs = model(inputs, None, None, mode='test')

			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()

		print('------> epoch: {} --> Test loss = {:.4f} Test Accuracy = {:.4f} '.format(epoch,test_loss / len(testloader.dataset), 100.*correct / len(testloader.dataset)))

	acc = 100.*correct/total
	if acc > best_acc:
		checkpoint(acc, epoch)
		best_acc = acc

	return best_acc


def checkpoint(acc, epoch):
	# Save checkpoint.
	print('Saving..')
	state = {
		'model': model.state_dict(),
		'optimizer' : optimizer.state_dict(),
		'acc': acc,
		'epoch': epoch,
		'seed' : args.manualSeed
	}
   
	torch.save(state, args.save_dir + 'checkpoint.t7')

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 at 600 and 900 epochs"""
    lr = args.lr_
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
		train(epoch)
		best_accuracy = test(epoch)
	
	print('Best Accuracy = ', best_accuracy)

