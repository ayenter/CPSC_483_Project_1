# ~~~ Author: Tom ~~~

# === imports ===

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, linear_model


# === code ===

# --- file names ---
file_name_predictors = "Data/predictors.npy"
file_name_train_predictors = "Data/train_indexes.npy"
file_name_test_predictors = "Data/test_indexes.npy"
file_name_labels_color = "Data/labels_color.npy"
file_name_labels_quality = "Data/labels_quality.npy"
file_name_labels_quality_binary = "Data/labels_quality_binary.npy"


# --- load data ---
X = np.load(file_name_predictors)
y_color = np.load(file_name_labels_color)
y_quality = np.load(file_name_labels_quality)
y_quality_binary = np.load(file_name_labels_quality_binary)

train_indexes = np.load(file_name_train_predictors)
test_indexes = np.load(file_name_test_predictors)


# --- split training and testing ---
X_train = X[train_indexes,:]
X_test = X[test_indexes,:]


# --- FUNCTION: get best number of neighbors using squared error ---
# --- get_best_n_neighbors( [int], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray ) ---



# --- fit classifier ---
for y in [[y_color, "Color"] , [y_quality, "Quality"], [y_quality_binary, "Binary Quality"]]:
	y_train = y[0][train_indexes]
	y_test = y[0][test_indexes]
	n_neighbors = get_best_n_neighbors(range(1,30,2), y_train, y_test, X_train, X_test)
	print (y[1] + " Neighbors: " + str(n_neighbors))
	clf = linear_model.LinearRegression()
	clf.fit(X_train, y_train)
	print (y[1] + " Squared Error: " + str(((clf.predict(X_test)-y_test)**2).sum()))
	plt.scatter(y_test,clf.predict(X_test))
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.title(y[1])
	plt.show()
