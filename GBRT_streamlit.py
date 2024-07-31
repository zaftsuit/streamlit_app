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
st.title(' Prediction for Patients with sTCSCI  ')


# In[3]:


st.header('Demographic')
#2.Age
Age= st.number_input(label='Age(y)')
#1.Sex
Sex=st.radio(label='Sex',options=['Male','Female'])


# In[4]:


st.header('Injury')
#4.ISS
ISS=st.slider(label='Injury severity score')
#12 Time of injury
Time_injury=st.slider(label='Time of injury')
#5 Organs damage 
Organs_damage=st.radio(label='Organs damage',options=['Zero','Single','multiple'])
#6 Cervical fracture
Cervical_fracture=st.radio(label='Cervical fracture',options=['Non-fractures','Upper(C1-2)','Lower(C3-7)'])
#10.NLI
NLI=st.radio(label='NLI',options=['C1-C4','C5-C8'])
#11.AIS grade
AIS_grade=st.radio(label='AIS Grade',options=['A','B'])


# In[5]:


st.header('Comorbidity')
#3 CCI
CCI=st.slider(label='Charlson Comorbidity Index')
#9 Underlying spinal diseases
Diseases=st.radio(label='Underlying spinal diseases',
                   options=['None','Cervical Spondylosis','AS','OPLL'])


# In[6]:


st.header('Information of treatment')
#13 Surgical method
Surgical_method=st.radio(label='Surgical Method',
                   options=['Conservative','Anterior','Posterior','Combined'])
#14 Syrgery timing
Syrgery_timing=st.radio(label='Surgery Timing',
                   options=['Non-surgery','Early-term','Medium-term','Later-term'])
#15 Blood
Blood_transfusion= st.number_input(label='Blood Transfusion')
#16 Nourishment
Nourishment=st.radio(label='Nourishment',
                   options=['Normal','Enteral','Parenteral'])


# In[7]:


st.header('Perioperative management')
#17 Complications
Complications=st.radio(label='Complications',
                   options=['Non-complications','Respiratory failure','Pneumonia','Urinary tract infection','Deep vein thrombosis'])
#8 HLOS
HLOS=st.number_input(label='HLOS(d)')
#7 Intensive_care_rate
Intensive_care_rate=st.number_input(label='Intensive care rate')


# In[8]:


#采集数据
if st.button("Predict"):
    # Unpickle classifier
    GBRT = joblib.load("GBRT.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[Sex,Age,CCI,ISS,Organs_damage,Cervical_fracture,Intensive_care_rate,
                    HLOS,Diseases,NLI,AIS_grade,Time_injury,Surgical_method,
                    Syrgery_timing,Blood_transfusion,Nourishment,Complications]], 
                    columns = ['Sex', 'Age', 'CCI','ISS','Organs damage','Cervical fracture','Intensive care rate',
                                'HLOS','Underlying spinal diseases','NLI','AIS grade','Time of injury','Surgical method',
                                'Syrgery timing','Blood transfusion','Nourishment','Complications'])
    X = X.replace(["A", "B"], [1, 0])
    X = X.replace(["C1-C4", "C5-C8"], [1, 0])
    X = X.replace(['Male','Female'], [0, 1])
    X = X.replace(['Zero','Single','multiple'], [0,1,2])
    X = X.replace(['None','Cervical Spondylosis','AS','OPLL'], [0,1,2,3])
    X = X.replace(['Non-fractures','Upper(C1-2)','Lower(C3-7)'], [0,1,2])
    X = X.replace(['Non-surgery','Early-term','Medium-term','Later-term'], [0,1,2,3])
    X = X.replace(['Conservative','Anterior','Posterior','Combined'], [0,1,2,3])
    X = X.replace(['Normal','Enteral','Parenteral'], [0,1,2])
    X = X.replace(['Non-complications','Respiratory failure','Pneumonia','Urinary tract infection','Deep vein thrombosis'], [0,1,2,3,4])
    #结果
    def survival_time(model,patient):
        va_times=np.arange(0,120)
        chf_funcs=model.predict_cumulative_hazard_function(patient)
        Time=()
        for fn in chf_funcs:#
            if fn(va_times[-1])<0.5:#在最后的预测时间内死亡全部累计概率不到0.6
                time_value=999
                Time=('This patient had no predicted death for 60 months')
                return Time
            else:
                for time in va_times:
                    if fn(time)>=0.5:
                        time_value=time#发生结局的最短时间
                        break
                Time=('The prognosis survival time of the patients was expected to be {} months'.format(time)) 
                return Time
    prediction = GBRT.predict(X)[0]
    patient = X[X.index==0]
    ST = survival_time(GBRT,patient)
    
    def risk_groups(model,patient):
        y_risk=model.predict(patient)
        group=()
        for fn in y_risk:#
            if fn<1.9:
                group=('Low-risk group')
                return group
            if fn>=1.9:
                group=('High-risk group')
                return group 
    #预测死亡时间
    patient = X[X.index==0]
    rg=risk_groups(GBRT,patient)
    
    # Output prediction
    st.header('outcome prediction')
    st.text(f"mortality risk:{rg}")
    st.text(f"Predicting Outcomes:{ST}")


# In[ ]:




