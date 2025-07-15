# water-potability1
```python
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import missingno as msno 
from warnings import simplefilter 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix 
simplefilter("ignore") 
data = pd.read_csv('/kaggle/input/water-potability/water_potability.csv') 
data.head()

data_count = data.isnull().sum()
print(data_count[0:10])
print(f'The dataset has {data.isna().sum().sum()} null values.')
data.describe()
