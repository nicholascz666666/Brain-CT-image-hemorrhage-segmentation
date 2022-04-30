import torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import os
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np	
Backslash = "//"
from skimage.transform import resize
torch.set_printoptions(threshold=np.inf)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
all_cate = ["epidural","intraparenchymal","intraventricular","multi","normal","subdural"]
class CTdataset(Dataset):
    def __init__(self, ypath, img_path,window_type = 2):
        #get all y lable csv
        ypath = Path(ypath)
        img_path = Path(img_path)
        self.ylable = pd.read_csv(ypath)
        self.table = {}
        self.table["tensor_matrix"] = {}
        self.table["file_name"] = {}
        self.table["category"] = {}
        #set the wanted window_type
        if window_type == 0:
            self.window_type = "brain_bone_window"
        elif window_type == 1:
            self.window_type = "brain_window"
        elif window_type == 2:
            self.window_type = "max_contrast_window"
        else:
            self.window_type = "subdural_window"
        
        # transfor data and store it in the table
        for  cat in all_cate:
            #get path
            path = img_path / cat / cat / self.window_type 
            for image_name in os.listdir(path):
                #read image
                single_img_path = path / image_name
                single_image = Image.open(single_img_path)
                matrix = transforms.ToTensor()(single_image).numpy()
                imgResized = resize(matrix, (128, 128,3))  
                self.table["tensor_matrix"][image_name] = imgResized.flatten()
                self.table["file_name"][image_name] = image_name
        
        # iterate ylable and find relevant information
        all_file = self.table["file_name"]
        for index, row in self.ylable.iterrows():
            if row["Image"] + ".jpg" in all_file:
                # normal image
                if row['any'] == 0:
                   self.table["category"][row["Image"] + ".jpg"] = -1
                   continue
                # multiple
                if sum(row[1:]) > 2:
                   self.table["category"][row["Image"] + ".jpg"] = 10  
                   continue
                # epidural = 0     
                if row['epidural'] == 1:
                    self.table["category"][row["Image"] + ".jpg"] = 0
                # intraparenchymal = 1
                if row['intraparenchymal'] == 1:
                    self.table["category"][row["Image"] + ".jpg"] = 1
                # intraventricular = 2
                if row['intraventricular'] == 1:
                    self.table["category"][row["Image"] + ".jpg"] = 2
                # subarachnoid = 3
                if row['subarachnoid'] == 1:
                    self.table["category"][row["Image"] + ".jpg"] = 3
                # subdural = 4
                if row['subdural'] == 1:
                    self.table["category"][row["Image"] + ".jpg"] = 4
             

    def get_table(self):
        return self.table

dataset = CTdataset(
    ypath= r'/home/cheng.zhi1/project_4570/project/hemorrhage-labels.csv',
    img_path = r"/home/cheng.zhi1/project_4570/project/renders")          

table = pd.DataFrame(dataset.get_table())
table.drop(columns=["file_name"])
table.to_csv('all_picture_matrix.csv')
print("completed save csv")
del dataset


x = table['tensor_matrix']
y = table['category']



param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
import numpy as np
print(np.array(y_test))
print(accuracy_score(y_pred,y_test))