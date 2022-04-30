from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

#predefine parameters

control = [0.1,1,10]

gamma = [0.001,0.1,1]

kernel = ['rbf','poly']

df = pd.read_csv(r"mixed_windows.csv")
df = df.drop(columns= ['Unnamed: 0'])
df = df.astype({"Target": object})

for index, row in df.iterrows():

  if row['Target'] == 0:

    df.at[index,'Target'] = 'normal'

  if row['Target'] == 10:

    df.at[index,'Target'] = 'multi'

  if row['Target'] == 1:

    df.at[index,'Target'] = 'epidural'

  if row['Target'] == 2:

    df.at[index,'Target'] = 'intraparenchymal'

  if row['Target'] == 3:

    df.at[index,'Target'] = 'intraventricular'   

  if row['Target'] == 4:

    df.at[index,'Target'] = 'subarachnoid'

  if row['Target'] == 5:

    df.at[index,'Target'] = 'subdural'
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# param_grid={'C':[0.1,1,10],'gamma':[0.001,0.1,1],'kernel':['rbf','poly']}

# svc=svm.SVC(probability=True)

# model=GridSearchCV(svc,param_grid)

 

#split data into training and testing sets

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

print(f'Splitted Successfully')

print(f'x_train shape: {x_train.shape}')

 

# model.fit(x_train,y_train)

# print('The Model is trained well with the given images')

# y_pred=model.predict(x_test)

# print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

# print(f"best parameters: {model.best_params_}")

#create svm model

 

keys =[]

# lof_scores = []

# lof_sMean = []

# lof_scoreSTD = []

accuracy = []

for k in kernel:

  for g in gamma:

    for c in control:

      clf = svm.SVC(kernel=k, C=c, gamma=g, random_state=42)

      clf.fit(x_train, y_train)

      y_pred = clf.predict(x_test)

      accu = metrics.accuracy_score(y_test, y_pred)

      accuracy.append(accu)

      key = f"{k},{g},{c}"

      keys.append(key)

      print(f"kernal: {k : <5}, gamma: {g : <6}, C: {c : < 4}, Accuracy: {accu}")

 

max_accuracy = max(accuracy)

max_index = accuracy.index(max_accuracy)

 

k, g, c = keys[max_index].split(",")

print(f"kernal: {str(k) : <5}, gamma: {str(g) : <6}, C: {str(c) : <4}")

clf = svm.SVC(kernel=k, C=float(c), gamma=float(g), random_state=42)

scores = cross_val_score(clf, x, y, cv=5)

print("performance %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()) )