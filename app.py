#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# In[2]:


st.title('Prediction for Patients with severe TCSCI')


# In[ ]:


#1.Age
Age= st.number_input(label='Age(y)',min_value=15, max_value=99)
#2 CCI
CCI=st.slider(label='Charlson Comorbidity Index',min_value=1, max_value=10)
#3.ISS
ISS=st.slider(label='Injury severity score',min_value=15, max_value=75)
#4 Thoracic and abdominal organs damage
Thoracic_abdominal_organs_damage=st.radio(label='Thoracic abdominal organs damage',options=['Non-damage','Single','multiple'])
#5 Cervical fracture
Cervical_fracture=st.radio(label='Cervical fracture',options=['Non-fractures','Upper(C1-2)','Lower(C3-7)'])
#6.NLI
NLI=st.radio(label='Neurological level of impairment',options=['C1-C4','C5-C8'])
#7 Time of injury
Time_injury=st.slider(label='Time of injury',min_value=1, max_value=10)
#8 Surgery timing
Surgical_timing=st.radio(label='Surgical Timing',
                   options=['Non-surgery','Early(＜3d)','Delay(≥3d)'])
#9.Transfusion
Transfusion=st.radio(label='Transfusion',options=['No-surgery','Transfusion','No-transfusion'])
#10.Critical care
ventilator_support=st.radio(label='Ventilator support',options=['No','Yes'])
#11. Nourishment
Nourishment=st.radio(label='Nourishment',
                   options=['Normal','Enteral','Parenteral'])
#12.Sex
Sex=st.radio(label='Sex',options=['Male','Female'])
#13.Energy of damage
Energy_of_damage=st.radio(label='Energy of damage',options=['High','Low'])
#14.AISA grade
AISA_grade=st.radio(label='AISA grade',options=['A','B'])
#15 Complications
Complications=st.radio(label='Complications',options=['Non-complications','pneumonia','Failure of organs'])


# In[ ]:


#collection
if st.button("Predict"):
    # Unpickle classifier
    RSF = joblib.load("RSF1.pkl")
    data=[
    [0,0,1,0,5,0,0,0,0,0,1,21,0,2,16],[1,0,1,1,3,2,0,2,0,0,1,30,0,2,25],[1,0,1,1,2,2,1,1,2,0,2,58,0,1,30],[1,1,1,1,7,1,1,2,0,1,2,25,2,2,35],
          [1,1,1,0,2,2,1,0,2,0,1,57,0,2,33],[1,0,1,1,1,1,0,2,2,0,1,50,0,2,41],[0,0,1,1,1,2,0,2,1,1,1,46,2,1,26],[1,1,1,1,1,0,1,2,1,0,1,48,2,2,26],
          [1,0,1,0,1,2,0,0,0,0,1,35,0,2,25],[0,1,1,1,5,2,1,2,2,1,2,57,1,2,30],[0,0,1,1,1,2,0,2,1,0,1,49,2,1,25],[0,0,1,1,1,2,1,2,2,0,1,57,0,1,25],
          [1,1,0,1,2,0,1,1,2,0,1,59,0,1,17],[0,0,1,0,3,2,1,1,2,0,1,51,0,2,30],[1,0,1,0,1,2,0,0,4,0,2,56,0,1,35],[1,0,1,0,1,2,0,0,2,0,2,58,0,2,33],
          [0,0,1,0,6,0,1,0,0,1,2,40,0,2,24],[1,0,1,0,1,0,1,1,0,0,2,33,0,2,25],[0,0,1,0,1,0,0,0,3,0,1,66,2,1,24],[1,0,1,0,1,2,1,1,2,0,1,46,2,2,25],
          [1,0,1,1,2,2,0,2,0,0,2,23,0,1,34],[1,0,1,1,3,2,0,2,0,1,1,33,0,2,34],[1,0,1,1,1,1,1,1,3,2,2,59,1,2,38],[1,1,1,1,1,0,0,1,0,0,2,32,0,1,26],
          [0,0,1,1,1,2,0,1,1,1,1,45,0,1,21],[0,0,1,1,3,2,0,2,2,0,2,43,1,2,38],[1,0,1,0,5,1,1,0,1,1,1,43,0,2,25],[0,0,1,0,1,2,1,1,0,0,1,31,2,1,25],
          [0,0,1,1,1,2,1,2,1,1,1,47,0,2,29],[1,0,1,1,5,2,1,2,0,0,1,33,0,2,43],[1,0,1,0,3,1,0,1,2,2,1,60,1,2,36],[1,0,1,1,1,2,1,2,1,2,1,45,2,2,35],
          [1,0,1,0,3,0,0,0,3,2,0,59,2,0,25],[1,0,1,1,1,2,1,2,0,1,1,39,1,2,42],[1,0,1,1,1,2,0,2,1,1,1,49,0,2,21],[0,0,1,1,1,2,1,2,1,1,2,50,1,2,42],
          [1,1,1,1,3,1,1,2,0,1,2,29,0,2,33],[0,0,1,1,1,0,1,1,0,0,2,39,2,1,25],[1,0,1,1,2,2,1,2,0,1,1,20,0,2,38],[1,0,1,0,3,1,1,0,4,1,0,70,1,0,25],
          [0,0,1,1,2,2,1,2,2,1,1,50,2,2,38],[1,0,1,0,4,1,1,1,2,0,1,58,0,2,29],[0,0,1,1,7,2,1,0,4,1,0,79,2,0,30],[0,0,1,1,2,2,1,2,3,2,1,65,1,2,42],
          [0,0,1,0,1,0,0,0,1,0,1,43,2,2,16],[0,0,0,0,1,2,0,0,4,0,2,56,0,2,17],[1,0,1,0,5,1,1,0,3,0,1,45,0,2,16],[0,0,1,0,2,2,0,1,4,0,1,63,0,1,26],
          [1,0,1,0,6,2,1,1,5,1,2,63,2,2,34],[0,0,0,0,1,2,0,1,3,1,1,61,0,2,15],[1,0,1,1,5,2,0,0,1,0,1,46,2,2,34],[1,0,1,1,2,2,1,1,2,0,2,65,0,2,25],
          [0,0,1,1,1,2,1,2,2,0,0,56,0,0,30],[1,0,1,1,1,2,1,2,2,1,1,52,0,1,29],[1,1,1,0,1,2,0,1,2,0,2,33,0,1,33],[0,0,1,1,5,2,0,1,3,0,1,54,2,1,38],
          [0,0,1,0,1,2,1,1,1,0,1,40,0,2,29],[1,0,1,1,1,1,1,2,0,1,2,39,0,1,42],[1,0,1,1,3,2,0,2,1,0,1,44,0,2,21],[1,0,1,0,1,2,1,2,3,0,2,70,0,1,40],
          [1,0,1,1,3,2,1,1,1,1,1,45,2,2,29],[1,0,1,0,1,1,0,0,0,1,1,21,1,2,25],[1,0,1,0,1,0,1,0,1,0,1,60,0,2,25],[0,0,1,1,1,2,0,1,1,0,1,49,2,1,25],
          [1,0,1,0,7,2,0,1,0,1,1,39,2,2,34],[1,0,1,1,4,2,1,2,1,0,1,41,0,2,25],[0,0,0,0,1,0,1,2,0,2,1,37,0,2,21],[1,1,1,1,2,0,0,2,0,2,1,28,1,2,25],
          [0,0,1,0,3,2,1,1,3,2,0,61,0,0,38],[1,0,1,1,1,2,1,2,0,0,1,37,0,2,25],[1,0,1,0,2,2,1,1,0,1,0,20,0,0,20],[1,0,1,1,2,2,1,1,0,0,1,22,0,1,25],
          [1,0,1,1,1,2,1,1,0,0,2,21,2,1,29],[1,0,1,0,4,2,1,2,2,2,1,56,0,2,24],[1,0,1,0,1,2,1,1,0,0,0,15,0,0,16],[1,0,1,0,1,2,0,0,4,0,0,67,0,0,27],
          [1,0,1,1,1,2,0,2,1,0,1,36,2,1,26],[1,0,1,0,1,2,1,0,2,0,0,51,0,0,25],[0,0,1,0,1,0,1,1,2,0,2,60,2,1,25],[1,1,1,1,1,1,0,1,5,0,1,69,0,1,25],
          [0,0,1,1,1,2,0,1,1,1,1,52,0,1,30],[1,0,1,1,5,1,0,0,1,1,1,48,2,2,18],[1,0,1,1,3,0,1,1,3,1,1,62,0,2,26],[0,0,1,0,1,2,1,2,0,1,1,31,0,2,38],
          [0,0,1,1,4,2,0,2,1,1,2,54,0,2,38],[1,0,1,1,1,2,0,1,1,0,1,49,2,1,30],[0,0,1,1,4,2,1,2,2,1,1,51,0,2,29],[1,0,1,0,1,1,0,1,0,0,1,25,0,2,25],
          [1,0,1,1,8,2,0,0,1,0,1,42,0,2,35],[1,0,1,1,1,2,1,1,3,0,1,63,2,2,25],[1,0,1,1,8,2,1,2,4,1,0,65,1,0,42],[0,0,1,1,1,2,0,1,1,0,1,43,2,1,25],
          [1,0,1,0,2,2,0,2,1,0,1,43,0,2,33],[1,1,0,1,1,2,1,0,2,0,1,51,0,1,16],[1,0,1,1,2,2,0,2,1,1,1,46,0,1,25],[1,1,1,1,1,1,1,2,2,2,0,48,1,0,45],
          [1,0,1,1,3,1,0,2,1,2,1,45,0,2,25],[1,0,0,1,1,2,0,2,3,0,2,63,0,2,16],[1,0,1,0,1,0,1,0,3,0,0,67,0,0,27],[0,0,1,1,1,2,0,1,1,0,1,42,0,2,18],
          [0,0,1,1,1,0,0,1,2,0,1,51,0,2,17],[1,1,1,1,1,2,1,1,3,0,1,65,0,1,26],[0,0,1,1,6,2,1,2,4,2,2,68,0,2,30],[0,0,1,1,1,2,1,2,4,2,0,56,0,0,45],
          [0,0,1,1,1,2,1,1,2,1,1,57,2,2,38],[0,0,1,1,1,2,0,1,2,1,2,60,2,2,34],[1,0,1,1,1,2,0,2,3,1,1,53,0,2,21],[1,0,1,1,4,1,1,2,2,1,1,51,1,2,29],
          [1,1,1,1,6,2,1,2,2,1,1,55,2,2,21],[0,0,1,1,2,2,0,1,1,0,1,44,0,1,26],[1,0,1,1,2,1,1,2,3,1,1,61,0,2,29],[1,0,1,1,1,2,0,2,3,0,1,57,1,2,29],
          [0,0,0,0,1,2,0,0,1,0,1,48,0,1,17],[1,0,1,1,1,1,0,1,3,0,1,60,0,2,29],[1,0,1,0,8,0,1,1,1,0,1,48,0,2,16],[0,0,1,0,7,2,1,0,0,0,1,35,0,2,34],
          [0,0,1,1,1,2,1,1,3,0,1,60,2,2,29],[0,0,1,1,1,2,1,1,0,0,1,28,0,1,27],[1,0,1,1,4,2,1,2,2,2,0,47,1,0,51],[0,0,1,1,2,0,0,2,3,0,1,67,0,2,25],
          [1,0,1,1,3,2,1,1,1,1,0,75,2,0,25],[1,0,1,1,1,2,1,2,1,2,0,42,0,0,42],[0,0,1,1,1,2,1,1,1,1,1,49,2,2,25],[0,1,1,0,1,2,0,1,1,1,1,43,2,1,20],
          [1,0,1,1,1,1,1,2,2,2,2,50,2,2,40],[1,0,1,0,4,0,0,0,0,0,1,27,0,2,16],[1,0,1,1,3,2,1,2,5,1,2,68,0,2,42],[1,0,1,1,2,2,1,2,0,0,1,37,0,1,38],
          [1,0,1,1,4,2,1,0,2,0,1,60,0,2,18],[1,1,1,1,1,0,1,1,1,0,2,46,2,2,27],[1,0,1,0,3,2,1,1,0,0,2,37,1,2,16],[1,0,1,1,8,2,0,2,3,0,1,43,2,2,34],
          [1,0,1,1,1,2,0,2,1,0,1,33,0,1,25],[1,0,1,1,1,1,1,2,3,1,0,67,1,0,50],[1,0,1,1,2,1,1,1,0,0,1,33,0,2,26],[1,1,1,1,1,0,1,1,0,0,1,19,2,1,25],
          [1,0,1,1,3,2,1,2,0,1,1,24,0,2,32],[1,0,1,1,1,1,0,2,0,1,1,35,0,1,27],[0,0,1,1,1,2,1,0,3,0,0,63,0,0,29],[1,0,1,0,2,0,1,0,4,0,0,61,2,0,27],
          [0,1,1,1,1,2,1,2,2,0,1,47,2,1,30],[0,0,0,0,1,2,0,1,0,0,2,24,2,2,17],[0,0,1,0,3,2,1,0,2,0,1,52,0,2,16],[0,0,1,1,1,2,0,2,2,1,1,58,0,2,26],
          [0,0,0,0,4,2,0,0,0,0,2,29,0,2,16],[1,1,0,1,4,0,0,1,4,0,1,68,0,2,20],[0,0,1,0,1,2,0,0,3,0,1,60,1,2,15],[0,0,0,0,2,2,1,0,5,1,2,61,0,1,21],
          [0,0,1,1,1,2,0,2,0,0,1,27,0,2,35],[1,0,1,1,1,2,0,2,2,0,2,61,0,1,35],[1,0,1,1,2,1,0,2,3,0,1,68,2,2,33],[1,1,1,1,1,2,1,2,0,0,1,24,2,2,34],
          [0,0,1,0,1,2,0,0,0,0,1,25,0,2,20],[1,0,1,0,1,2,1,0,0,0,2,30,0,1,30],[1,0,1,1,1,0,0,2,2,1,1,60,0,1,25],[0,0,0,1,6,0,1,1,2,0,1,54,0,2,16],
          [1,0,1,1,1,0,1,2,1,0,2,48,2,1,25],[1,0,1,1,3,2,1,2,5,1,1,71,2,1,29],[0,0,1,0,1,0,0,0,0,0,1,27,0,2,17],[1,0,1,1,8,1,1,2,2,2,1,52,2,2,50],
          [0,0,1,0,3,2,1,1,1,0,0,44,2,0,26],[0,0,1,1,1,0,1,1,1,0,1,47,0,1,26],[0,0,1,1,2,2,0,1,2,1,1,59,0,2,26],[0,0,1,1,1,2,0,1,3,0,1,52,0,1,29],
          [1,0,1,1,1,2,0,2,0,0,2,22,0,1,34],[0,0,1,0,1,2,0,0,2,1,1,49,0,2,21],[1,0,1,1,7,2,0,2,4,0,1,53,0,2,35],[1,1,0,0,3,2,0,0,1,0,1,46,0,2,16],
          [0,0,1,0,1,2,1,0,1,0,1,41,0,2,25],[0,0,1,0,1,2,1,2,0,2,0,23,0,0,26],[0,0,1,1,1,2,1,1,1,2,0,44,0,0,40],[1,1,1,1,5,0,1,1,1,0,1,46,0,2,25],
          [0,0,1,1,2,2,1,1,1,0,1,41,0,2,16],[1,0,1,1,6,2,0,1,2,1,1,52,2,2,30],[0,1,1,0,4,0,1,1,0,0,0,38,0,0,27],[0,1,1,0,5,2,0,0,1,0,1,46,0,2,27],
          [1,0,1,1,1,2,1,1,4,1,1,65,0,1,35],[1,0,1,1,1,2,0,1,3,1,1,62,0,1,38],[0,0,1,1,4,2,1,2,4,0,1,63,2,2,29],[1,0,1,1,1,2,1,2,1,1,1,48,0,2,18],
          [1,0,0,1,2,0,0,0,1,1,1,47,0,1,20],[1,0,0,1,1,2,0,2,0,1,1,29,2,2,17],[1,0,1,1,1,1,1,2,3,2,2,66,0,2,34],[1,0,1,1,2,0,1,1,2,0,0,60,2,0,25],
          [1,0,1,1,1,0,0,1,2,0,1,58,0,2,25],[0,1,0,0,1,2,0,0,2,0,1,58,0,1,16],[1,0,1,1,1,2,1,2,1,1,1,44,0,2,36],[1,0,1,1,7,2,1,1,0,1,2,35,0,2,21],
          [1,0,0,0,1,2,1,1,0,0,1,38,0,2,16],[1,0,1,1,1,0,0,1,2,0,1,51,0,1,25],[1,0,1,1,2,2,0,1,0,1,1,38,2,2,35],[1,0,1,1,1,2,1,2,3,2,1,61,0,1,41],
          [1,0,1,0,6,2,0,0,1,1,1,43,0,2,36],[1,0,0,1,2,2,1,1,3,0,1,63,0,2,16],[1,0,1,1,2,2,0,1,0,0,2,28,2,2,34],[0,0,1,1,1,2,1,2,1,0,2,44,2,2,26],
          [0,0,0,0,1,0,1,0,3,0,2,55,0,2,16],[1,0,1,1,7,1,1,2,1,1,1,45,2,2,33],[1,0,1,1,5,2,0,1,0,1,1,45,2,2,36],[0,0,1,1,2,2,0,2,2,1,2,58,0,2,38],
          [1,0,1,1,6,2,1,2,0,1,1,44,0,2,38],[0,0,0,0,1,2,0,0,4,1,1,68,0,1,17],[1,0,1,1,3,0,0,1,2,0,1,57,0,2,25],[0,0,1,1,3,0,0,2,0,1,1,36,2,2,24],
          [0,0,1,1,1,2,0,1,1,0,1,47,0,1,26],[1,0,1,0,1,2,1,0,0,0,0,40,2,0,26],[0,0,1,1,2,2,1,1,2,0,2,57,0,2,20],[1,0,1,1,4,2,1,2,3,1,1,55,0,2,42],
          [0,1,1,1,1,2,0,2,2,0,2,62,0,2,35],[1,0,1,0,1,2,0,0,0,0,2,30,0,2,20],[1,0,1,0,4,0,0,0,0,1,0,34,1,0,30],[1,0,1,1,2,2,0,0,2,0,0,45,2,0,33],
          [1,0,1,1,1,2,1,1,1,0,0,41,0,0,27],[0,0,1,1,1,2,1,1,1,0,2,42,0,1,26],[1,1,0,1,1,0,0,1,1,0,1,46,0,1,21],[0,0,1,0,1,2,1,1,1,1,2,42,1,1,38],
          [1,0,1,1,3,1,1,2,2,1,1,55,0,2,38],[1,0,0,1,7,0,1,1,3,1,1,61,0,2,20],[1,0,1,1,1,2,0,1,0,1,1,20,0,1,29],[1,0,1,0,1,2,1,0,1,0,0,50,0,0,31],
          [0,1,1,0,1,2,0,1,0,1,2,35,0,2,20],[0,0,1,0,2,2,1,1,1,0,1,46,0,1,26],[1,0,1,1,1,1,1,2,2,0,2,48,0,1,34],[1,0,1,0,1,2,0,0,0,1,1,29,0,1,25],
          [1,0,1,1,7,2,0,1,2,0,1,52,2,2,30],[0,0,0,1,1,0,0,1,3,1,2,65,0,2,17],[1,0,1,0,2,2,1,1,2,0,1,57,2,2,26],[1,0,1,0,5,0,0,1,0,0,1,25,0,2,24],
          [0,0,1,1,3,0,0,0,3,0,1,65,2,2,25],[0,0,1,0,2,2,0,2,0,1,1,22,0,2,29],[0,0,1,1,1,0,0,2,3,0,2,64,0,1,26],[0,0,1,1,1,2,0,1,3,1,1,55,0,1,25],
          [1,0,1,1,1,1,0,2,0,1,2,32,2,2,35],[0,0,1,0,6,2,0,0,1,0,1,46,0,2,17],[1,0,1,1,3,0,1,1,0,1,2,33,0,2,24],[0,1,1,1,1,2,1,2,2,1,0,49,2,0,42],
          [0,0,1,0,1,2,0,0,4,0,1,73,2,2,25],[1,0,1,1,1,2,0,1,1,0,2,42,0,2,33],[1,0,1,1,2,0,0,1,4,0,2,64,0,2,21],[1,0,1,1,3,2,1,1,0,1,0,15,0,0,34],
          [0,0,1,1,1,2,1,2,0,0,1,28,0,1,25],[1,1,1,1,2,2,1,2,1,1,1,49,2,2,41],[1,0,1,1,1,2,1,0,1,1,1,49,0,1,29],[0,0,1,0,1,0,0,1,4,1,1,66,0,1,20],
          [1,1,1,1,2,1,1,2,1,1,2,49,2,1,35],[0,1,1,1,7,2,0,1,2,0,1,55,0,2,26],[0,0,1,0,1,0,0,0,2,0,1,58,0,2,16],[1,1,1,1,1,2,1,1,2,1,1,54,0,1,35],
          [1,0,1,0,1,2,1,2,0,1,1,44,0,1,38],[0,0,1,1,4,2,1,1,1,0,0,42,2,0,35],[1,0,1,1,2,1,1,2,3,1,2,65,2,2,38],[1,0,1,1,1,1,1,2,2,1,1,52,2,1,42],
          [0,0,1,0,1,2,1,1,1,1,1,45,0,2,26],[0,0,1,1,3,2,1,1,0,1,1,32,0,2,34],[1,0,1,1,2,2,1,2,3,1,2,56,1,2,42],[1,0,1,0,2,2,0,0,2,1,2,48,2,2,38],
          [1,0,1,1,1,2,1,1,4,0,2,69,2,2,27],[0,0,1,0,1,2,1,1,0,0,1,33,2,2,30],[1,0,1,0,3,2,1,2,3,1,0,48,2,0,34],[1,0,1,1,1,2,1,2,1,2,2,41,2,2,41],
          [1,0,1,1,5,2,1,2,3,2,1,61,1,2,43],[1,0,0,0,5,0,0,0,1,0,2,50,2,2,16],[0,0,0,1,7,2,1,2,1,2,1,52,0,2,17],[0,0,1,1,4,2,0,1,0,0,1,41,0,2,21],
          [1,0,1,1,1,1,1,2,1,1,1,43,2,1,30],[0,0,1,0,1,2,0,0,0,0,1,33,0,2,25],[0,0,1,1,1,2,1,1,3,2,0,64,0,0,42],[1,0,1,1,1,2,1,2,1,0,1,42,2,1,45],
          [1,0,1,1,1,0,1,2,2,1,1,57,2,2,25],[1,0,1,1,1,2,1,2,3,0,1,59,0,1,21],[0,0,1,0,2,2,1,1,4,1,1,51,2,2,26],[0,0,1,0,1,2,0,0,0,0,0,40,0,0,20],
          [1,0,1,1,2,2,0,2,0,0,1,45,0,1,34],[0,0,1,0,1,0,0,0,2,0,1,50,0,2,16],[1,0,1,1,1,2,1,0,3,0,1,66,2,1,27],[1,0,1,1,1,2,0,1,0,0,1,36,0,1,25],
          [1,0,1,1,1,2,1,2,1,1,1,42,2,2,42],[1,0,1,1,4,2,1,1,3,0,1,51,2,2,27],[0,1,1,1,2,2,1,1,2,1,2,59,0,1,35],[1,0,1,1,1,2,1,1,0,0,2,47,1,1,43],
          [1,0,1,1,5,2,0,2,1,1,1,57,2,2,38],[0,0,1,1,3,2,0,1,0,0,2,23,0,2,26],[1,1,1,1,1,2,1,1,1,0,2,45,2,1,35],[1,0,1,1,2,2,0,1,1,1,0,30,2,0,35],
          [0,0,1,1,1,2,1,1,0,0,2,21,2,1,26],[0,0,1,1,2,2,1,0,2,1,1,52,2,2,29],[1,0,1,1,1,1,1,1,2,0,1,58,0,2,17],[0,0,1,0,1,2,0,0,0,1,1,37,0,2,16],
          [1,1,1,1,2,2,1,2,2,1,1,55,0,2,34],[1,0,1,1,8,2,0,2,2,1,0,53,2,0,35],[1,1,1,1,1,2,1,1,0,1,1,23,0,1,33],[0,0,1,0,3,2,0,0,0,0,1,33,0,2,17],
          [1,0,1,1,1,0,0,2,0,2,1,26,2,2,25],[1,0,1,1,1,2,1,1,2,0,1,51,1,2,26],[1,0,1,1,2,2,1,2,0,2,1,15,0,2,35],[1,0,1,1,1,2,0,2,4,0,1,66,0,1,26],
          [1,1,1,1,4,2,0,1,0,1,1,27,0,2,35],[0,0,1,0,5,2,1,2,2,2,0,60,0,0,27],[1,1,1,0,2,0,1,0,2,0,0,59,0,0,26],[0,0,1,0,1,2,0,1,6,0,1,70,0,1,27],
          [1,0,1,1,1,0,0,2,0,0,1,24,2,2,29],[1,0,1,1,1,2,1,2,0,1,1,39,0,2,29],[1,0,1,1,1,2,1,1,3,0,1,62,0,1,30],[0,0,1,1,1,2,0,1,0,0,1,38,2,2,20],
          [1,0,1,1,1,0,0,1,3,0,2,67,0,2,22],[1,1,0,0,1,0,0,2,1,0,0,47,0,0,18],[1,0,1,0,2,2,1,2,1,1,1,49,0,2,42],[1,1,1,0,2,2,1,1,1,0,1,46,0,2,26],
          [0,0,1,0,1,0,1,1,2,1,1,54,0,2,25],[0,0,1,1,1,2,1,1,2,1,1,59,0,2,25],[1,0,1,0,3,2,0,0,0,0,1,15,0,2,25],[1,0,0,1,3,2,0,1,2,0,1,54,2,2,20],
          [0,0,1,1,1,2,0,2,0,0,2,22,2,2,38],[0,0,1,1,1,2,1,2,2,0,1,54,0,1,26],[0,0,1,0,1,2,0,1,3,0,1,54,0,2,34],[1,1,1,0,4,1,0,0,0,1,1,19,0,2,25],
          [0,0,0,1,2,2,0,1,4,0,1,74,0,2,16],[0,0,1,1,1,2,1,2,1,1,1,41,2,2,29],[1,1,1,0,1,1,1,1,0,1,1,33,0,1,29],[1,0,0,1,1,1,0,2,2,0,2,57,0,2,20],
          [1,0,1,1,1,2,1,1,1,0,1,48,2,2,35],[1,1,1,1,1,0,1,0,3,0,1,67,0,1,26],[0,0,0,0,1,2,1,0,1,1,1,47,0,1,16],[0,0,1,1,1,2,1,2,0,0,1,31,2,1,25],
          [1,0,1,1,2,2,1,2,0,1,1,27,0,2,38],[1,0,1,0,8,1,1,0,3,0,1,60,0,2,17],[1,0,1,0,2,2,1,0,3,0,1,61,0,2,21],[0,0,1,0,1,1,0,0,1,0,1,43,0,2,16],
          [0,0,1,1,1,2,1,1,1,0,1,49,0,2,29],[1,0,1,1,8,2,0,1,2,1,1,60,0,2,16],[0,0,1,0,4,0,0,1,3,0,1,67,0,2,18],[1,0,1,0,1,0,0,0,2,0,1,60,0,2,25],
          [1,0,1,0,1,2,0,0,2,0,0,57,0,0,26],[1,1,1,0,1,2,1,1,2,0,2,49,2,1,29],[1,0,1,0,1,1,1,0,2,0,1,51,0,1,21],[0,0,1,1,1,0,1,1,1,0,1,44,2,1,27],
          [0,0,1,1,1,0,1,1,2,0,1,51,0,1,26],[0,0,1,0,5,2,0,2,2,0,1,59,1,2,34],[1,0,1,1,5,1,0,1,1,1,1,48,1,2,27],[0,1,0,0,6,2,0,0,0,0,2,32,0,2,16],
          [1,0,1,0,1,0,0,0,3,0,1,66,0,2,17],[1,1,1,1,2,2,1,2,0,0,2,40,1,2,26],[1,0,1,1,2,2,1,2,0,1,2,41,1,2,42],[0,0,1,0,5,2,0,0,3,0,0,65,2,0,16],
          [0,0,1,0,1,2,0,1,2,1,1,55,0,1,30],[1,0,1,1,1,2,1,2,0,1,1,40,0,1,42],[0,0,1,0,1,2,0,0,1,1,2,46,0,1,21],[1,0,1,1,1,1,1,1,2,1,1,59,1,1,45],
          [1,0,1,1,1,2,0,1,1,2,1,49,2,1,30],[1,0,1,0,1,2,0,1,0,0,1,32,0,1,25],[1,0,0,1,1,0,0,0,2,0,1,60,0,1,17],[0,0,1,1,2,2,0,1,1,0,1,48,2,1,26],
          [0,0,1,0,3,2,1,0,0,0,1,25,0,2,25]]
    X_train= pd.DataFrame(data=data,columns = ['NLI', 'Sex','Energy of damage','Ventilator support','Time of injury',
                              'Cervical fracture','AISA grade','Nourishment','CCI','Thoracic and abdominal organs damage',
                               'Transfusion','Age','Complications','Surgical timing','ISS'])
    explainer = shap.Explainer(RSF.predict,X_train)
    # Store inputs into dataframe
    X= pd.DataFrame([[NLI,Sex,Energy_of_damage,ventilator_support,Time_injury,
                      Cervical_fracture,AISA_grade,Nourishment,CCI,Thoracic_abdominal_organs_damage,
                      Transfusion,Age,Complications,Surgical_timing,ISS]], 
                    columns = ['NLI', 'Sex','Energy of damage','Ventilator support','Time of injury',
                              'Cervical fracture','AISA grade','Nourishment','CCI','Thoracic and abdominal organs damage',
                               'Transfusion','Age','Complications','Surgical timing','ISS'])
    X= X.replace(['C1-C4', 'C5-C8'], [1,0])
    X= X.replace(['Yes', 'No'], [1,0])
    X= X.replace(['Non-damage','Single','multiple'], [0,1,2])
    X= X.replace(['Non-fractures','Upper(C1-2)','Lower(C3-7)'], [0,1,2])
    X= X.replace(['Non-surgery','Early(＜3d)','Delay(≥3d)'], [0,1,2])
    X= X.replace(['Normal','Enteral','Parenteral'], [0,1,2])
    X= X.replace(['No-surgery','Transfusion','No-transfusion'],[0,1,2])
    X= X.replace(['Male','Female'], [0, 1])
    X= X.replace(['High','Low'], [1, 0])
    X= X.replace(['A','B'], [1, 0])
    X= X.replace(['Non-complications','pneumonia','Failure of organs'], [0,1,2])
    #survival time
    def survival_time(model,patient):
        va_times=np.arange(0,60)
        chf_funcs=model.predict_cumulative_hazard_function(patient)
        Time=()
        for fn in chf_funcs:#
            if fn(va_times[-1])<0.5:
                time_value=999
                Time=('According to our model,the survival time of the patient\nis expected to be more then 60 months')
                return Time
            else:
                for time in va_times:
                    if fn(time)>=0.5:
                        time_value=time
                        break
                Time=('According to our model, the survival time of the patient\nis expected to be {} months'.format(time)) 
                return Time
    prediction = RSF.predict(X)[0]
    patient = X[X.index==0]
    ST = survival_time(RSF,patient)
  
    #risk-group
    def risk_groups(model,patient):
        y_risk=model.predict(patient)
        group=()
        for fn in y_risk:#
            if fn<29.2:
                group=('Low-risk group')
                return group
            if fn>=29.2:
                group=('High-risk group')
                return group 
    #final
    patient = X[X.index==0]
    rg=risk_groups(RSF,patient)
  
    p1=plt.figure()
    shap_values1=explainer(X)
    shap.plots.waterfall(shap_values1[0])
    plt.savefig("shap_waterfall.png", bbox_inches='tight', dpi=1200)
  
    p2=plt.figure()
    shap_values = explainer(patient)
    shap.plots.force(shap_values,matplotlib=True,show=False,contribution_threshold=0.01)
    plt.savefig("shap_force.png", bbox_inches='tight', dpi=1200)
    # Output prediction
    st.header('outcome prediction')
    st.text(f"mortality risk:\n{rg}")
    st.text(f"Predicting Outcomes:\n{ST}")
    st.text(f"Risk indicators plot：\n")
    st.image("shap_waterfall.png")
    st.image("shap_force.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




