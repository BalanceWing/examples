import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F 

import numpy as np
from pathlib import Path 
import requests
import pickle
import gzip

# get and process data 
def get_data (train_ds, valid_ds, bs):
	return (
		DataLoader(train_ds, batch_size = bs, shuffle = True),
		DataLoader(valid_ds, batch_size = bs * 2))


class WrappedDataLoader:
	"""docstring for WrappedDataLoader"""
	def __init__(self, dl, func):		
		self.dl = dl
		self.func = func

	def __len__ (self):
		return len (self.dl)

	def __iter__(self):
		batches = iter (self.dl)
		for b in batches :
			yield (self.func(*b))

# Sequential
class Lambda(nn.Module):
	"""docstring for Lambda"""
	def __init__(self, func):
		super().__init__()
		self.func = func

	def forward (self, x):
		return self.func (x)

def preprocess (x, y):
	return x.view (-1, 1, 28, 28), y

# train data
def loss_batch (model, loss_func, xb, yb, opt = None):
	loss = loss_func (model (xb), yb)

	if opt is not None:
		loss.backward ()
		opt.step ()
		opt.zero_grad ()

	return loss.item(), len(xb)

def fit (epochs, model, loss_func, opt, train_dl, valid_dl):
	for epoch in range (epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch (model, loss_func, xb, yb, opt)

		model.eval ()
		with torch.no_grad ():
			losses, nums = zip (*[loss_batch (model, loss_func, xb, yb) for xb, yb in valid_dl])

		#statistics
		val_loss = np.sum (np.multiply (losses, nums)) / np.sum(nums)

		print (epoch, val_loss)




#setup mnist data
DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'

PATH.mkdir (parents = True, exist_ok = True)
URL = "http://deeplearning.net/data/mnist"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists ():
	content = requests.get (URL + FILENAME).content
	(PATH / FILENAME).open("wb").write(content)

# open dataset
with gzip.open ((PATH / FILENAME).as_posix(), "rb" ) as f :
	((x_train, y_train), (x_valid, y_valid), _) = pickle.load (f, encoding = "latin-1")



##### training loop
bs = 64 # batch size
lr = 0.5 # learning rate
epochs = 10 

x_train, y_train, x_valid, y_valid = map ( torch.tensor, (x_train, y_train, x_valid, y_valid))

train_ds = TensorDataset (x_train, y_train)
valid_ds = TensorDataset (x_valid, y_valid)

train_dl, valid_dl = get_data (train_ds, valid_ds, bs)
train_dl = WrappedDataLoader (train_dl, preprocess)
valid_dl = WrappedDataLoader (valid_dl, preprocess)

model = nn.Sequential (
	nn.Conv2d (1, 16, kernel_size = 3, stride = 2, padding = 1),
	nn.ReLU (),
	nn.Conv2d (16, 16, kernel_size = 3, stride = 2, padding = 1),
	nn.ReLU (),
	nn.Conv2d (16, 10, kernel_size = 3, stride = 2, padding = 1),
	nn.ReLU(),
	nn.AdaptiveAvgPool2d (1),
	Lambda (lambda x : x.view (x.size(0), -1)),
	)

loss_func = F.cross_entropy

opt = optim.SGD (model.parameters(), lr = lr, momentum = 0.9)

fit (epochs, model, loss_func, opt, train_dl, valid_dl)
