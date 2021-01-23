import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch.utils.data as Data
import visdom
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import statistics
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler

def normalizedata(data):
	num_features = data.shape[1]

	mean = np.array([data[:,j].mean() for j in range(num_features)]).reshape(num_features)
	std = np.array([data[:,j].std() for j in range(num_features)]).reshape(num_features)

	for i in range(num_features):
		if float(std[i]) != 0:
			data[:, i] = (data[:, i] - float(mean[i])) * (1 / float(std[i]))
		else:
			data[:, i] = np.ones((data.shape[0]))
	return data

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
		self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

	def forward(self, x):
		x = F.relu(self.hidden(x))      # activation function for hidden layer
		x = self.predict(x)             # linear output
		return x



def train(x, y, xtest, ytest):
	torch.manual_seed(1)
	num_features = x.shape[1]
	num_output = 1
	learning_rate = 0.01
	BATCH_SIZE = 64
	EPOCH = 10


	x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor)), Variable(torch.from_numpy(y).type(torch.FloatTensor))
	torch_dataset = Data.TensorDataset(x, y)

	loader = Data.DataLoader(
		dataset=torch_dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, num_workers=2,)

	net = Net(n_feature= num_features, n_hidden=10, n_output=num_output)     # define the network
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.2)
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	my_images = []
	fig, ax = plt.subplots(figsize=(16,10))
	#vis = visdom.Visdom()
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

	# start training

	for epoch in range(EPOCH):
		for step, (batch_x, batch_y) in enumerate(loader): # for each training step

			b_x = Variable(batch_x)
			b_y = Variable(batch_y)
			b_y = b_y.view(b_y.size()[0], 1)

			prediction = net(b_x)     # input x and predict based on x
			loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)


			optimizer.zero_grad()   # clear gradients for next train
			loss.backward()         # backpropagation, compute gradients
			optimizer.step()        # apply gradients

		'''epoch_loss = loss
		loss_window = vis.line(
			Y=torch.zeros((1)).cpu(),
			X=torch.zeros((1)).cpu(),
			opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['Loss']))
		vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([epoch_loss]).unsqueeze(0).cpu(),win=loss_window,update='append')'''
		if epoch % 100 == 0:
			#print('Loss at step {0}: {1}'.format(epoch, loss))
			pass
		exp_lr_scheduler.step()


	xtest, ytest = Variable(torch.from_numpy(xtest).type(torch.FloatTensor)), Variable(torch.from_numpy(ytest).type(torch.FloatTensor))
	ytest = ytest.view(ytest.size()[0], 1)
	prediction = net(xtest)     # input x and predict based on x
	loss = loss_func(prediction, ytest)
	print('Validation loss {0}'.format(loss)) 
	print('Validation r2score {0}'.format(r2_score(ytest.data.numpy(), prediction.data.numpy())))
	return loss.data.numpy(), r2_score(ytest.data.numpy(), prediction.data.numpy())


def regression(X, Y):
	
	X = normalizedata(X)
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

	loss_ = []
	r2sc = []


	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		lss, r2s = train(X_train, Y_train, X_test, Y_test)
		r2sc.append(float(r2s))
		loss_.append(float(lss))

	print('Test Loss mean:', statistics.mean(loss_))
	print('std loss:', statistics.stdev(loss_))      

	print('Test r2score mean:', statistics.mean(r2sc))
	print('std r2score:', statistics.stdev(r2sc))          
	 
