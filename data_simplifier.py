import numpy as np


file_name_red = "winequality-red.csv"
file_name_white = "winequality-white.csv"

file_name_output = "winequality-full.npy"

dataset1 = np.loadtxt(open(file_name_red,"rb"), delimiter=";",skiprows=1)
dataset2 = np.loadtxt(open(file_name_white,"rb"), delimiter=";",skiprows=1)

print("Red Wine Dataset shape: " + str(dataset1.shape))
print("White Wine Dataset shape: " + str(dataset2.shape))

dataset1 = np.append(dataset1, np.zeros((len(dataset1),1)),1)
dataset2 = np.append(dataset2, np.ones((len(dataset2),1)),1)

print("Red Wine Dataset reshaped: " + str(dataset1.shape))
print("White Wine Dataset reshaped: " + str(dataset2.shape))

dataset3 = np.append(dataset1, dataset2, 0)

print("Full Wine Dataset shape: " + str(dataset3.shape))

np.save(file_name_output, dataset3)

# np.load(file_name_output)