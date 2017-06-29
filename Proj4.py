
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[5]:


df_pima=pd.read_csv('C:\\Users\\Naqsh\\Downloads\\asd\\diabetes.csv', names=['Pragnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])


# In[25]:


df_pima.describe()


# In[26]:


X_features = pd.DataFrame(data = df_pima, columns = ["Glucose","BMI","Age"])           
X_features.head(2)
#Considering the 3 features showing the max correlation. 


# In[27]:


Y = df_pima.iloc[:,8]
Y.head(3)


# In[28]:


from sklearn.model_selection import train_test_split               #Split Data: Training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.25, random_state=10)


# In[29]:


from sklearn.model_selection import KFold                          #Classification Models
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
models = []

models.append(("Logistic Regression:",LogisticRegression()))
models.append(("Naive Bayes:",GaussianNB()))
models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=3)))
models.append(("Decision Tree:",DecisionTreeClassifier()))
models.append(("Support Vector Machine-linear:",SVC(kernel="linear",C=0.2)))
models.append(("Support Vector Machine-rbf:",SVC(kernel="rbf")))
models.append(("Ranom Forest:",RandomForestClassifier(n_estimators=5)))

print('Models appended...')


# In[30]:


results = []           #Results
names = []
for name,model in models:
    kfold = KFold(n_splits=5, random_state=3)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)


# In[ ]:




