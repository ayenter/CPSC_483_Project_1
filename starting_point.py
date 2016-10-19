# === imports ===

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


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


# *** FOR KNN ONLY ***
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
# *** ------------ ***


# --- fit classifier ---
for y in [[y_color, "Color"] , [y_quality, "Quality"], [y_quality_binary, "Binary Quality"]]:
	y_train = y[0][train_indexes]
	y_test = y[0][test_indexes]

	# *** FOR KNN ONLY ***
	# n_neighbors = get_best_n_neighbors(range(1,30,2), y_train, y_test, X_train, X_test)
	# print (y[1] + " Neighbors: " + str(n_neighbors))
	# *** ------------ ***
	
	clf = # SCIKIT-LEARN ALGORITHM
	clf.fit(X_train, y_train)
	
	print (y[1] + " Squared Error: " + str(((clf.predict(X_test)-y_test)**2).sum()))
	
	plt.scatter(y_test,clf.predict(X_test))
	
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	
	plt.title(y[1])
	
	plt.show()








