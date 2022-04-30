import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderFolderPath = r'E:\project\renders'
#'brain_bone_window',
listOfWindows = ['brain_bone_window', 'brain_window', 'max_contrast_window', 'subdural_window']
hemorrhage_CSV_PATH = r'E:\project\hemorrhage-labels.csv'
def load_type_window_Image_To_DF(type_window_PATH, window, dataframe):
  for entry in os.listdir(type_window_PATH):
    image_path = type_window_PATH+ '/' + entry
    # read image
    imgArray = imread(image_path)
    # resize image to 150*150*3
    imgResized = resize(imgArray, (150, 150,3))
    # grayscale image so that we have 150*150
    grayImage = rgb2gray(imgResized)
    # flattern the image matrix
    fattern_image = list(grayImage.flatten())
    imageIndex = entry[0:-4]
    dataframe.at[imageIndex, window] = fattern_image
  print('Finished loading images from', type_window_PATH)

def load_All_Type_window_To_CSV(window):
  df = pd.read_csv(hemorrhage_CSV_PATH)
  df = df.set_index('Image')
  df[window] = ''
  for entry in os.listdir(renderFolderPath):
    window_path = renderFolderPath + '/' + entry +'/' + entry + '/' + window
    load_type_window_Image_To_DF(window_path, window, df)
  df = df.reset_index()
  df.to_csv(window + '.csv', header=True, mode='a')

for w in listOfWindows:
  load_All_Type_window_To_CSV(w)
  print('Finished saving window: ' + w + ' to ' +  w+ '.CSV')