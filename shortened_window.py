import numpy as np
import pandas as pd
import os
mixed_window = pd.read_csv('max_contrast_window.csv',nrows =150)
mixed_window.to_csv("shortened _max_contrast_windows.csv")