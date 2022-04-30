import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from skimage.color import rgb2gray
import os
brain_bone_window = pd.read_csv('brain_bone_window.csv', low_memory=False)
brain_window = pd.read_csv('brain_window.csv', low_memory=False)
max_contrast_window = pd.read_csv('max_contrast_window.csv', low_memory=False)
subdural_window = pd.read_csv('subdural_window.csv', low_memory=False)

index = ['Image','Unnamed: 0', 'any', 'epidural', 'intraparenchymal', 'intraventricular',
       'subarachnoid', 'subdural']
all = brain_bone_window.set_index(index).join(brain_window.set_index(index)).join(max_contrast_window.set_index(index)).join(subdural_window.set_index(index))
all = all.dropna()
print(len(all))
all.head()
all = all.reset_index()
testDF = all.drop(columns=['Unnamed: 0'])
print(all.dtypes)
print(all.columns)
print(testDF.dtypes)
print(testDF.columns)
def convertTypeToNum (dataframe):
  for index, row in dataframe.iterrows():
    if row['any'] == 0 :
      dataframe.loc[index,'type'] = 0
    elif row['epidural'] + row['intraparenchymal'] + row['intraventricular']+ row['subdural'] > 1:
      dataframe.loc[index,'type'] = 10
    elif row['epidural'] == 1:
      dataframe.loc[index, 'type'] = 1
    elif row['intraparenchymal'] == 1:
      dataframe.loc[index, 'type'] = 2
    elif row['any'] == 1 & row['intraventricular'] == 1:
      dataframe.loc[index,'type'] = 3
    elif row['any'] == 1 & row['subarachnoid'] == 1:
      dataframe.loc[index,'type'] = 4
    elif row['any'] == 1 & row['subdural'] == 1:
      dataframe.loc[index,'type'] = 5
    else: dataframe.loc[index,'type'] = -10

convertTypeToNum(testDF)
testDF.head()
testDF = testDF.drop(columns=['any', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid','subdural' ])
testDF.head()
flat_data_arr = []
target_arr = []
for index, row in testDF.iterrows():
  bbw = np.fromstring(row['brain_bone_window'][1:-1], dtype=float, sep=',')
  bw = np.fromstring(row['brain_window'][1:-1], dtype=float, sep=',')
  mw = np.fromstring(row['max_contrast_window'][1:-1], dtype=float, sep=',')
  sw = np.fromstring(row['subdural_window'][1:-1], dtype=float, sep=',')
  c1 = np.add(bbw, bw)
  c2 = np.add(mw, sw)
  final = np.add(c1, c2)
  flat_data_arr.append(final)
  target_arr.append(row['type'])

flate_data = np.array(flat_data_arr)
target = np.array(target_arr)
newDF = pd.DataFrame(flate_data)
newDF['Target'] = target
newDF.to_csv("mixed_windows.csv")