import pandas as pd

dataset1 = pd.read_csv('mixed_windows.csv', nrows=2000)
dataset1.to_csv("dataset2_1000.csv")

dataset2 = pd.read_csv('mixed_windows.csv', nrows=5000)
dataset2.to_csv("dataset2_5000.csv")

dataset3 = pd.read_csv('mixed_windows.csv', nrows=10000)
dataset3.to_csv("dataset2_10000.csv")
