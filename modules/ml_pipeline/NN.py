import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from multilabelClassificationMetrics import metrics_precision_recall
import statistics
from sklearn.model_selection import KFold



def construct_matrix(samples):
	return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(labels):
	return np.array(labels).reshape(labels.shape[0], 1)


def readdata():
	train_dataset, train_labels = readmnist.trainadata()
	return construct_matrix(train_dataset), construct_labels(train_labels)


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



def relu_activation(data_array):
	return np.maximum(data_array, 0)

def softmax(output_array):
	logits_exp = np.exp((output_array))
	return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)


def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
	indices = np.argmax(y_onehot, axis = 1).astype(int)
	predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
	log_preds = np.log(predicted_probability)
	loss = -1.0 * np.sum(log_preds) / len(log_preds)
	return loss

def regularization_L2_softmax_loss(reg_lambda, weight1, weight2):
	weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
	weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
	return weight1_loss + weight2_loss

def ROCurve(testy, output_probs):
	# calculate scores
	lr_auc = roc_auc_score(testy, output_probs, multi_class = 'ovr')
	# summarize scores
	print('NN: ROC AUC=%.3f' % (lr_auc))
	metrics_precision_recall(testy, output_probs, 0, 4, 'NN')
	# calculate roc curves
	'''lr_fpr, lr_tpr, _ = roc_curve(testy, output_probs)
	# plot the roc curve for the model
	pyplot.plot(lr_fpr, lr_tpr, marker='.', label='MLP')
	# axis labels
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	# show the plot
	pyplot.show()'''
	return lr_auc

def train_NN(train_X, train_y, test_X, test_Y):
	training_data = train_X
	training_labels = train_y

	hidden_nodes = 15
	num_labels = training_labels.shape[1]
	num_features = training_data.shape[1]
	learning_rate = .01
	reg_lambda = .01
	batch_size = 100

	# Weights and Bias Arrays, just like in Tensorflow
	layer1_weights_array = np.random.normal(0, 1, [num_features, hidden_nodes]) 
	layer2_weights_array = np.random.normal(0, 1, [hidden_nodes, num_labels]) 

	layer1_biases_array = np.zeros((1, hidden_nodes))
	layer2_biases_array = np.zeros((1, num_labels))


	for step in range(500000):

		idx = np.random.randint(train_X.shape[0], size=batch_size)
		training_data = train_X[idx, :]
		training_labels = train_y[idx, :]

		input_layer = np.dot(training_data, layer1_weights_array)
		hidden_layer = relu_activation(input_layer + layer1_biases_array)
		output_layer = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
		output_probs = softmax(output_layer)
		
		loss = cross_entropy_softmax_loss_array(output_probs, training_labels)
		loss += regularization_L2_softmax_loss(reg_lambda, layer1_weights_array, layer2_weights_array)

		y_preds = np.argmax(output_probs, axis=1)
		y_true = np.argmax(training_labels, axis = 1)      

		output_error_signal = (output_probs - training_labels) / output_probs.shape[0]
		
		error_signal_hidden = np.dot(output_error_signal, layer2_weights_array.T) 
		error_signal_hidden[hidden_layer <= 0] = 0
		
		gradient_layer2_weights = np.dot(hidden_layer.T, output_error_signal)
		gradient_layer2_bias = np.sum(output_error_signal, axis = 0, keepdims = True)
		
		gradient_layer1_weights = np.dot(training_data.T, error_signal_hidden)
		gradient_layer1_bias = np.sum(error_signal_hidden, axis = 0, keepdims = True)

		gradient_layer2_weights += reg_lambda * layer2_weights_array
		gradient_layer1_weights += reg_lambda * layer1_weights_array

		layer1_weights_array -= learning_rate * gradient_layer1_weights
		layer1_biases_array -= learning_rate * gradient_layer1_bias
		layer2_weights_array -= learning_rate * gradient_layer2_weights
		layer2_biases_array -= learning_rate * gradient_layer2_bias
		

		if step % 100000 == 0:
				print('Loss at step {0}: {1}'.format(step, loss))
		if step % 200000 ==0:
			learning_rate = .001





	input_layer = np.dot(test_X, layer1_weights_array)
	hidden_layer = relu_activation(input_layer + layer1_biases_array)
	output_layer = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
	output_probs = softmax(output_layer)


	y_preds = np.argmax(output_probs, axis=1)
	y_true = np.argmax(test_Y, axis = 1)
		
	loss = cross_entropy_softmax_loss_array(output_probs, test_Y)
	loss += regularization_L2_softmax_loss(reg_lambda, layer1_weights_array, layer2_weights_array)
	accuracy = accuracy_score(y_true, y_preds)
   
	print(loss)
	print('validation accuracy:', accuracy_score(y_true, y_preds))
	auc = ROCurve(test_Y, output_probs)
	print(auc)
	return loss, accuracy, auc




def classification(X, Y):


	X = normalizedata(X)


	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

	accuracy_ = []
	loss_ = []
	auc_ = []

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		try:
			lss, acc, auc = train_NN(X_train, Y_train, X_test, Y_test)
		except:
			pass
		loss_.append(lss)
		accuracy_.append(acc)
		auc_.append(auc)


	print('test loss mean', statistics.mean(loss_))
	print('test loss stdev:', statistics.stdev(loss_))
	print('test acuracy:', statistics.mean(accuracy_))
	print('test accuracy stedv:', statistics.stdev(accuracy_))
	print('test auc mean:', statistics.mean(auc_))
	print('test auc stdev:', statistics.stdev(auc_))

