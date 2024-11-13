#!/usr/bin/env python
# coding: utf-8

# In[6]:


#ANS2
import numpy as np

def fn(msg):
    str_array = np.array(strings)
    result = [s[::-1] for s in str_array if len(s[::-1]) >= 5]  
    return result

msg = ["My", "name", "is", "Sushweta", "Bhattacharya", "hehe"]
ans = fn(msg)
print("The result is:", ans)


# In[38]:


#ANS 1:
from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, cross_val_score #, GridSearchCv
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC

#part a
data = load_breast_cancer()
df = pd.DataFrame(data = data.data, columns=data.feature_names)
df['target'] = data.target
print("PART A \n")
print("10 rows are ", df.head(10))
print("PART B \n")
print(df.describe().loc[['mean', 'std', 'min', 'max']])

#part c
print("Part C \n")
missing = df. isnull().sum()
print("The missing values are :", missing)


#split dataset part 4
X_train, X_test, y_train, y_test = train_test_split(df, df.target, test_size=0.3, random_state=42)


#part 5
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#part 7
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)




# In[26]:





# In[ ]:





# In[ ]:




