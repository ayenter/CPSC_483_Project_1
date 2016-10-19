import numpy as np


file_name_red = "winequality-red.csv"
file_name_white = "winequality-white.csv"

file_name_predictors = "predictors.npy"
file_name_labels_color = "labels_color.npy"
file_name_labels_quality = "labels_quality.npy"
file_name_labels_quality_binary = "labels_quality_binary.npy"

dataset1 = np.loadtxt(open(file_name_red,"rb"), delimiter=";",skiprows=1)
dataset2 = np.loadtxt(open(file_name_white,"rb"), delimiter=";",skiprows=1)

target_color = np.append(np.zeros((len(dataset1),1)), np.ones((len(dataset2),1)), 0)

target_quality = np.append(dataset1[:,len(dataset1[0])-1], dataset2[:,len(dataset2[0])-1], 0)

list_bi_quality = []
for i in target_quality:
	if i <= 5:
		list_bi_quality.append(0)
	else:
		list_bi_quality.append(1)

target_quality_binary = np.asarray(list_bi_quality)

dataset = np.append(np.delete(dataset1, len(dataset1[0])-1, 1), np.delete(dataset2, len(dataset2[0])-1, 1), 0)

np.save(file_name_predictors, dataset)
np.save(file_name_labels_color, target_color)
np.save(file_name_labels_quality, target_quality)
np.save(file_name_labels_quality_binary, target_quality_binary)