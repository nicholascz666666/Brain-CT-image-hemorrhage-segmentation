import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import datatable as dt
import polars

f = open("demosvm.txt", "a")
f.write("model for svm")
brain_bone_window = pd.read_csv('brain_bone_window.csv',low_memory=False)
print("reading finished")
brain_window = pd.read_csv('brain_window.csv',low_memory=False)
print("reading finished")
max_contrast_window = pd.read_csv('max_contrast_window.csv',low_memory=False)
print("reading finished")
subdural_window = pd.read_csv('subdural_window.csv',low_memory=False)
print("reading finished")

index = ['Image','Unnamed: 0', 'any', 'epidural', 'intraparenchymal', 'intraventricular',
       'subarachnoid', 'subdural']
all = brain_bone_window.set_index(index).join(brain_window.set_index(index)).join(max_contrast_window.set_index(index)).join(subdural_window.set_index(index))
all = all.dropna()


all = all.reset_index()
testDF = all.drop(columns=['Unnamed: 0'])

def convertTypeToNum (dataframe):
  for index, row in dataframe.iterrows():
    if row['any'] == 0 :
      dataframe.loc[index,'type'] = 'normal'#0
    elif row['epidural'] + row['intraparenchymal'] + row['intraventricular']+ row['subdural'] > 1:
      dataframe.loc[index,'type'] = 'multi' #10
    elif row['epidural'] == 1:
      dataframe.loc[index, 'type'] = 'epidural'#1
    elif row['intraparenchymal'] == 1:
      dataframe.loc[index, 'type'] = 'intraparenchymal'#2
    elif row['any'] == 1 & row['intraventricular'] == 1:
      dataframe.loc[index,'type'] = 'intraventricular' #3
    elif row['any'] == 1 & row['subarachnoid'] == 1:
      dataframe.loc[index,'type'] = 'subarachnoid'#4
    elif row['any'] == 1 & row['subdural'] == 1:
      dataframe.loc[index,'type'] = 'subdural'#5
    else: dataframe.loc[index,'type'] = '-10'

convertTypeToNum(testDF)


testDF = testDF.drop(columns=['any', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid','subdural' ])


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

x = newDF.iloc[:,:-1]
y = newDF.iloc[:,-1]

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score

#predefine parameters
control = [0.1,1,10]
gamma = [0.001,0.1,1]
kernel = ['rbf','poly']

#split data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print(f'Splitted Successfully')
print(f'x_train shape: {x_train.shape}')
f.write("x_train.shape:")
f.write("{},{}".format(x_train.shape[0],x_train.shape[1]))

#create svm model

keys =[]
lof_scores = []
lof_sMean = []
lof_scoreSTD = []
for k in kernel:
  for g in gamma:
    for c in control:
      clf = svm.SVC(kernel=k, C=c, gamma=g, random_state=42)
      scores = cross_val_score(clf, x, y, cv=5)
      key = f"kernal: {k}, gamma: {g}, C: {c}"
      keys.append(key)
      lof_scores.append(scores)
      lof_sMean.append(scores.mean())
      lof_scoreSTD.append(scores.std())
      print(key, "performance %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()) )
      f.write("performance %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

f.write("keys")
f.write(keys)
f.write("lof_scores")
for i in lof_scores:
  f.write(str(i))
f.write("lof_sMean")
for i in lof_sMean:
  f.write(str(i))
f.write("lof_scoreSTD")
for i in lof_scoreSTD:
  f.write(str(i))
f.close()