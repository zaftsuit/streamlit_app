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
                   options=['Non-surgery','Early','Delay'])
#9.Transfusion
Transfusion=st.radio(label='Transfusion',options=['No-surgery','Transfusion','No-transfusion'])
#10.Critical care
Critical_care=st.radio(label='Critical care',options=['No','Yes'])
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
    X_train = pd.read_csv('X_train.csv', low_memory=False)
    explainer = shap.Explainer(RSF1.predict,X_train)
    # Store inputs into dataframe
    X = pd.DataFrame([[NLI,Sex,Energy_of_damage,Critical_care,Time_injury,
                      Cervical_fracture,AISA_grade,Nourishment,CCI,Thoracic_abdominal_organs_damage,
                      Transfusion,Age,Complications,Surgical_timing,ISS]], 
                    columns = ['NLI', 'Sex','Energy of damage','Critical care','Time of injury',
                              'Cervical fracture','AISA grade','Nourishment','CCI','Thoracic and abdominal organs damage',
                               'Transfusion','Age','Complications','Surgical timing','ISS'])
    X = X.replace(["C1-C4", "C5-C8"], [1, 0])
    X = X.replace(["Yes", "No"], [1, 0])
    X = X.replace(['Non-damage','Single','multiple'], [0,1,2])
    X = X.replace(['Non-fractures','Upper(C1-2)','Lower(C3-7)'], [0,1,2])
    X = X.replace(['Non-surgery','Early(＜3d)','Delay(≥3d)'], [0,1,2])
    X = X.replace(['Normal','Enteral','Parenteral'], [0,1,2])
    X = X.replace(['No-surgery','Transfusion','No-transfusion'],[0,1,2])
    X = X.replace(['Male','Female'], [0, 1])
    X = X.replace(['High','Low'], [1, 0])
    X = X.replace(['A','B'], [1, 0])
    X = X.replace(['Non-complications','pneumonia','Failure of organs'], [0,1,2])
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
            if fn<20:
                group=('Low-risk group')
                return group
            if 20<=fn<45.5:
                group=('Medium-risk group')
                return group
            if fn>=45.5:
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




