#multilabelClassificationMetrics.py

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns
import matplotlib as mpl


mpl.rcParams.update(mpl.rcParamsDefault)
sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white') #sets the size of the charts

def metrics_precision_recall(Y_test, y_score, X_test, n_classes, classifier_name):

	#The average precision score in multi-label settings

	# For each class
	mpl.rcParams.update(mpl.rcParamsDefault)
	sns.set_style("white")
	sns.set_palette("Set2")
	plt.style.use('seaborn-white')
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
															y_score[:, i])
		average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
		y_score.ravel())
	average_precision["micro"] = average_precision_score(Y_test, y_score,
														 average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'
		  .format(average_precision["micro"]))

	#Plot the micro-averaged Precision-Recall curve
	plt.figure()
	plt.step(recall['micro'], precision['micro'], where='post')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(
		'Average precision score of' + classifier_name + ', micro-averaged over all classes: AP={0:0.2f}'
		.format(average_precision["micro"]))
	plt.savefig('precision-recall ' + classifier_name)


	#Plot Precision-Recall curve for each class and iso-f1 curves
	# setup plot details
	#colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

	plt.figure(figsize=(7, 8))
	f_scores = np.linspace(0.2, 0.8, num=4)
	lines = []
	labels = []
	for f_score in f_scores:
		x = np.linspace(0.01, 1)
		y = f_score * x / (2 * x - f_score)
		l, = plt.plot(x[y >= 0], y[y >= 0], alpha=0.2)
		plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

	lines.append(l)
	labels.append('iso-f1 curves')
	l, = plt.plot(recall["micro"], precision["micro"], lw=2)
	lines.append(l)
	labels.append('micro-average Precision-recall (area = {0:0.2f})'
				  ''.format(average_precision["micro"]))

	for i in range(n_classes):
		l, = plt.plot(recall[i], precision[i], lw=2)
		lines.append(l)
		labels.append('Precision-recall for class {0} (area = {1:0.2f})'
					  ''.format(i, average_precision[i]))

	fig = plt.gcf()
	fig.subplots_adjust(bottom=0.25)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(' Precision-Recall curve for multi-class' + classifier_name)
	plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
	plt.savefig('recall-precision ' + classifier_name)




