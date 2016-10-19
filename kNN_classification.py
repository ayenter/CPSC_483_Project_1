# === imports ===

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# === code ===

# --- file names ---
file_name_predictors = "Data/predictors.npy"
file_name_labels_color = "Data/labels_color.npy"
file_name_labels_quality = "Data/labels_quality.npy"
file_name_labels_quality_binary = "Data/labels_quality_binary.npy"

# --- load data ---
X = np.load(file_name_predictors)
y_color = np.load(file_name_labels_color)
y_quality = np.load(file_name_labels_quality)
y_quality_binary = np.load(file_name_labels_quality_binary)


n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf.fit(X, y_color)