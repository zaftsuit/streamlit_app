#!/usr/bin/env python
# coding: utf-8

# # BoostCox

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, GridSearchCV


# In[2]:


df=pd.read_csv(r"C:\Users\须臾\notebook\data\TCSCI_OK1.csv")


# In[3]:


X =df.drop(columns=['Disease survival','Survival months'])
X_dummy = pd.get_dummies(X,drop_first=True)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputer.fit(X_dummy)
X_imputed = imputer.transform(X_dummy)
X_imputed = pd.DataFrame(columns=X_dummy.columns, data=X_imputed)


# In[4]:


from sklearn.model_selection import train_test_split
y = df.loc[:,['Disease survival','Survival months']]
y[['Disease survival']] = y[['Disease survival']]=='Dead'
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=2023)


# In[5]:


from sksurv.util import Surv
y_train_ = Surv.from_dataframe(
    event='Disease survival', 
    time='Survival months', 
    data=y_train)
y_test_ = Surv.from_dataframe(
    event='Disease survival', 
    time='Survival months', 
    data=y_test)


# In[6]:


from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import numpy as np


# In[7]:


best_coxboost = GradientBoostingSurvivalAnalysis(loss="coxph",
                                                 learning_rate=0.05,
                                                 max_depth=2,
                                                  min_samples_leaf=5,
                                                 min_samples_split=2,
                                                 n_estimators=500)


# In[8]:


best_coxboost.fit(X_train, y_train_)


# In[9]:


#训练集
from sksurv.metrics import concordance_index_censored
y_pred = best_coxboost.predict(X_train)
result = concordance_index_censored(y_train_['Disease survival'], y_train_['Survival months'], y_pred)


# In[10]:


#测试集
from sksurv.metrics import concordance_index_censored
y_pred = best_coxboost.predict(X_test)
result = concordance_index_censored(y_test_['Disease survival'], y_test_['Survival months'], y_pred)


# In[11]:


import joblib


# In[12]:


joblib.dump(best_coxboost,'Gboost_cox.pkl')

