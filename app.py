#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# In[2]:


#大标题
st.title(' Severe tCSCI Survival Prediction App')


# In[3]:


st.header('Demographic')
#1.Age
Age= st.number_input(label='Age')


# In[4]:


st.header('Injury')
#2.ISS
ISS=st.slider(label='ISS')
#3.Injury level
Level=st.radio(label='NLI',options=['C1-C4','C5-C8'])
#4 Time of injury
Time_injury=st.slider(label='Time of injury')
#5.ASIA score
ASIA=st.radio(label='AIS Grade',options=['A','B'])
#6.Etiology
Etiology= st.selectbox(label='Etiology',
                       options=['Fall','MVA','Struck by Object','Sport-related','Others'])
#7.Damage energy 
Energy=st.radio(label='Damage energy',options=['High','Low'])






# In[5]:


st.header('Comorbidity')
#8 CCI
CCI=st.slider(label='CCI')
#9  Underlying spinal diseases
Diseases=st.radio(label='Underlying spinal diseases',
                   options=['None','Cervical Spondylosis','AS','OPLL'])


# In[6]:


st.header('Information of treatment')
#10 Treatment
Treatment=st.radio(label='Treatment',
                   options=['Surgery','Non-surgery'])
#11 Surgery time 
Time_surgery=st.radio(label='Syrgery Timing',
                   options=['Non-surgery','Early-term','Medium-term','Later-term'])
#12 Surgical method
Method=st.radio(label='Surgical Method',
                   options=['None','Anterior','Posterior','Combined'])
#13 Blood
Loss= st.number_input(label='Blood Loss')
Transfusion= st.number_input(label='Blood Transfusion')
#14 Nourishment
Nourishment=st.radio(label='Nourishment',
                   options=['None','Enteral','Parenteral'])


# In[7]:


st.header('Outcome')
#18 Injection
Infection=st.radio(label='Infection',options=['Yes','No'])
#15 HLOS
HLOS=st.number_input(label='HLOS')
#16 ICULOS
ICULOS=st.number_input(label='ICULOS')
#17 ICULOS/HLOS
ICULOS_HLOS=st.number_input(label='ICULOS/HLOS')


# In[8]:


#采集数据
if st.button("Predict"):
    
    # Unpickle classifier
    GBRT = joblib.load("GBRT.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[Age,ISS,Level,Time_injury,ASIA,Etiology,
                       Energy,CCI,Diseases,Treatment,Time_surgery,Method,
                      Loss,Transfusion,Nourishment,Infection,HLOS,ICULOS,ICULOS_HLOS]], 
                     columns = ['Age', 'ISS', 'NLI','Time of injury','AIS grade','Etiology',
                                'Energy of damage','CCI','Underlying spinal diseases','Treatment','Syrgery timing','Surgical method',
                                'Blood loss','Blood transfusion','Nourishment','Infection','HLOS','ICULOS','ICULOS/HLOS'
                               ])
    X = X.replace(["A", "B"], [1, 0])
    X = X.replace(["C1-C4", "C5-C8"], [1, 0])
    X = X.replace(['High','Low'], [1, 0])
    X = X.replace(['Fall','MVA','Struck by Object','Sport-related','Others'], [1,2,3,4,5])
    X = X.replace(['None','Cervical Spondylosis','AS','OPLL'], [0,1,2,3])
    X = X.replace(['Surgery','Non-surgery'], [1, 0])
    X = X.replace(['Non-surgery','Early-term','Medium-term','Later-term'], [0,1,2,3])
    X = X.replace(['None','Anterior','Posterior','Combined'], [0,1,2,3])
    X = X.replace(['None','Enteral','Parenteral'], [0,1,2])
    X = X.replace(['Yes','No'], [1, 0])
    # Get prediction
    p1=plt.figure(figsize=(16,5),dpi=150)
    ax1=p1.add_subplot(1,2,1)
    prediction = GBRT.predict(X)[0]
    funcs = GBRT.predict_survival_function(X)
    for fn in funcs:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.title('survival function accumulated over time')
    plt.ylabel("Survival probability")
    plt.xlabel("Time")
    plt.ylim(0, 1)
    
    ax2=p1.add_subplot(1,2,2)
    hazard=GBRT.predict_cumulative_hazard_function(X)
    for fm in hazard:
        plt.step(fm.x, fm(fm.x), where="post")
    plt.title('The hazard function accumulated over time')
    plt.ylabel("Cumulative Hazard")
    plt.xlabel("Time")
    plt.ylim(0, 1)
    plt.savefig('img.png')
    # Output prediction
    st.header('outcome prediction')
    st.text(f"mortality risk:{prediction}")
    st.image('img.png')
    


# In[ ]:





# In[ ]:




