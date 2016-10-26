# ~~~ Author: Bijan Hamidi ~~~

# === imports ===

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor


# === code ===

# --- file names ---
file_name_predictors = "Data/predictors.npy"
file_name_train_predictors = "Data/train_indexes.npy"
file_name_test_predictors = "Data/test_indexes.npy"
file_name_labels_color = "Data/labels_color.npy"
file_name_labels_quality = "Data/labels_quality.npy"
file_name_labels_quality_binary = "Data/labels_quality_binary.npy"


# --- load data ---
y_color = np.load(file_name_labels_color)
y_quality = np.load(file_name_labels_quality)
y_quality_binary = np.load(file_name_labels_quality_binary)

train_indexes = np.load(file_name_train_predictors)
test_indexes = np.load(file_name_test_predictors)

the_X = np.load(file_name_predictors)
the_X_w_color = np.column_stack((the_X, y_color))


# --- FUNCTION: get number from user ---
# --- get_valid_input(string) ---
def get_valid_input(prompt):
	while 1:
		try:
			return int(input(prompt))
		except ValueError:
			pass;

# --- FUNCTION: get best number of neighbors using squared error ---
# --- get_best_n_neighbors( [int], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray ) ---
def get_best_n_neighbors( possible_ns, a_y_train, a_y_test, a_X_train, a_X_test ):
	best_n_neighbors = 0
	best_sq_error = float('inf')
	for i in possible_ns:
		temp_n_neighbors = i
		temp_clf = neighbors.KNeighborsClassifier(temp_n_neighbors, weights="uniform", p=len(X[0]))
		temp_clf.fit(a_X_train, a_y_train)
		temp_sq_error = ((temp_clf.predict(a_X_test)-a_y_test)**2).sum()
		if temp_sq_error < best_sq_error:
			best_n_neighbors = temp_n_neighbors
			best_sq_error = temp_sq_error
	return best_n_neighbors;

# --- FUNCTION: add confusion matrrix to plot ---
# --- plot_confusion_matrix(numpy.ndarray (2D), [int/string], bool, string, plt.cm.Color) ---
# --- credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

# --- FUNCTION: add ROC graph to plot ---
# --- plot_ROC(int, numpy.ndarray (2D), numpy.ndarray (2D), string) ---
# --- credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_ROC(subplt, y_test, predicted, title):
    plt.figure(3)
    plt.subplot(subplt)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, predicted)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic for ' + title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

# --- input ---
the_k = get_valid_input("\nEnter k ('0' to exit): ")

# --- fit classifier ---
while the_k >= 1:
	plt.figure(1, figsize=(15,10))
	plt.figure(2, figsize=(15,10))
	plt.figure(3, figsize=(15,10))

	for z in [[y_color, the_X, "Color", 231, ["red", "white"]] , [y_quality, the_X, "Quality", 232], [y_quality, the_X_w_color, "Quality with Color as a Feature", 235], [y_quality_binary, the_X, "Binary Quality", 233, ["Low", "High"]], [y_quality_binary, the_X_w_color, "Binary Quality with Color as a Feature", 236, ["Low", "High"]]]:
		
		# --- set data variables ---
		n_neighbors = the_k
		y = z[0]
		X = z[1]
		title = z[2]
		plt_pos = z[3]

		# --- split training and testing ---
		X_train = X[train_indexes,:]
		X_test = X[test_indexes,:]

		y_train = z[0][train_indexes]
		y_test = z[0][test_indexes]

		# --- print info ---
		print ("\n --- " + title + " --- ")
		print ("(Approx. best k is " + str(get_best_n_neighbors(range(1,30,2), y_train, y_test, X_train, X_test)) + ")")

		# --- run algorithm ---
		clf = KNeighborsRegressor(n_neighbors, weights="uniform", p=len(X[0]))
		clf.fit(X_train, y_train)

		# --- results ---
		predicted = clf.predict(X_test)
		predicted_rounded = np.asarray([0 if x<0 else x for x in np.round(predicted)])

		# --- print error ---
		print ("Squared Error: " + str(((predicted-y_test)**2).sum()))
		print ("Accuracy Score: " + str(metrics.accuracy_score(y_test, predicted_rounded)))
		print ("R2 value: " + str(metrics.r2_score(y_test, predicted)))

		# --- plotting ---
		if len(z) == 5:
			labels = z[4]
		else:
			test_max = max( int(y_test.max()), int(predicted_rounded.max()) )
			test_min = min( int(y_test.min()), int(predicted_rounded.min()) )
			labels = range(test_min, test_max+1)
		plt.figure(1)
		plt.subplot(plt_pos)
		plt.scatter(y_test,predicted)
		plt.xlabel('Actual')
		plt.ylabel('Predicted')
		plt.title(title)
		plt.figure(2)
		plt.subplot(plt_pos)
		plot_confusion_matrix(confusion_matrix(y_test, predicted_rounded), classes=labels)

		# --- Logistic Regression only. Generate ROC graphs ---
		if title == "Color":
			plot_ROC(221, y_test, predicted, title)
		elif title == "Binary Quality":
			plot_ROC(222, y_test, predicted, title)
		elif title == "Binary Quality with Color as a Feature":
			plot_ROC(223, y_test, predicted, title)

	# --- display plot ---
	plt.show()

	# --- more input ---
	the_k = get_valid_input("\nEnter k ('0' to exit): ")

print ("")