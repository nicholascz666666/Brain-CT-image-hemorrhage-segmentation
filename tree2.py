import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Set device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


data = pd.read_csv(r"dataset_10000.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
f = open("demofile2.txt", "a")





f.write("model for RandomForestClassifier")
#RandomForestClassifier
table_test ={"accuracy_score":[],"f1_score":[],"roc_auc_score":[]}
decisions = [10,50,100,500]
for d in decisions:
  rfc = RandomForestClassifier(n_estimators=d,max_depth=11)
  rfc = rfc.fit(x_train,y_train)
  y_test_pred = rfc.predict(x_test)
  table_test["accuracy_score"].append(accuracy_score(y_test,y_test_pred))
  scores = cross_val_score(rfc, x, y, cv=5)
  f.write("performance %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

f.write(str(table_test))

#AdaBoostClassifier
f.write("model for AdaBoostClassifier")
table_test_ada ={"accuracy_score":[],"f1_score":[],"roc_auc_score":[]}
for d in decisions:
  rfc = AdaBoostClassifier(n_estimators=d,base_estimator=DecisionTreeClassifier())
  rfc = rfc.fit(x_train,y_train)
  y_test_pred = rfc.predict(x_test)
  
  table_test_ada["accuracy_score"].append(accuracy_score(y_test,y_test_pred))
  scores = cross_val_score(rfc, x, y, cv=5)
  f.write("performance %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

f.write(str(table_test_ada))
f.close()

