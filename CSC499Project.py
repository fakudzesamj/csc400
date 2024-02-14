import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

from predictionModel import predict

df = pd.read_csv('courses.csv')
ProgrammeCode=df['ProgrammeCode']
ProgrammeName=df['ProgrammeName']
mytext=""
#get the id of the menu item clicked
with st.sidebar:
    selected=option_menu(
        menu_title:="menu",
        options:=["Home","Predict Grad","Predict Course","About","Help","Contact"],
        index:=0
        )
if selected=="Home":
        st.title("Home")
if selected=="Predict Grad":
    st.title("Predict Graduation and CGPA")
    with st.form(key="form1"):
        c1,c2,c3=st.columns(3)
        c4,c5,c6,c7=st.columns(4)
        c8=st.columns(1)
        with c1:
            Sex=st.selectbox("Select Gender *",['Female','Male'])
        with c2:
            Age=st.slider(label ="Select Age Group *", min_value=16, max_value=60)
        with c3:
            Prog=st.selectbox("Select Programme of Study *",ProgrammeCode)
        with c4:
            Semester1=st.number_input("Enter Semester 1 GPA *",min_value=0 ,max_value=6,value=-1,step=1)
        with c5:
            Semester2=st.number_input("Enter Semester 2 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c6:    
            Semester3=st.number_input("Enter Semester 3 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c7:    
            Semester4=st.number_input("Enter Semester 4 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c4:   
            Semester5=st.number_input("Enter Semester 5 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c5:   
            Semester6=st.number_input("Enter Semester 6 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c6:  
            Semester7=st.number_input("Enter Semester 7 GPA",min_value=0,max_value=6,value=-1,step=1)
        with c7:  
            Semester8=st.number_input("Enter Semester 8 GPA",min_value=0,max_value=6,value=-1,step=1)
            if(Sex=="Male"):
                Gender=1
            if(Sex=="Female"):
                Gender=0
            if (Age<25):
                AgeGroup=0
            if (Age<35):
                AgeGroup=1
            if (Age>35):
                AgeGroup=2
        with c4:  
            grad=predict([Gender,AgeGroup,int(Prog),Semester1])
            if(grad[0]==0):
                mytext='Judging from your previous semesters, You are going to have chalenges in graduate on record time, you need to work a little bit harder to be able to graduate. With your current progression you are at resit of getting a CGPA of'+str(grad[1])
               
            else: 
                if(grad[1]==2):
                    mytext="Judging from your previous semesters, You have a chance of graduating on record time, you need to work a little bit harder to be able to archive a CGPA of better than "+str(grad[1])
                if(grad[1]>=3):
                    mytext="Judging from your previous semesters, You have a chance of graduating on record time, you are on course of getting CGPA of better than "+str(grad[1])+" Keep up the good work!!!"
                  
            submit =st.form_submit_button(label='Predicted')
        
        st.markdown(mytext)
if selected=="Predict Course":
        st.title("Lets Predict course")
if selected=="About":
        st.title("Lets Predict course")
if selected=="Contact":
        st.title("Lets Predict course")
