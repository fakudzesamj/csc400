import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from predictionModel import predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
df8 = pd.read_csv('courses.csv')
ProgrammeCode=df8['ProgrammeCode']

df= pd.read_csv('updateDataset.csv')
Y=df["GRADUATED"]
X=df[['Sex','Sponsored','AgeGroup','programme_code','Semester1','Semester2','Semester3',
'Semester4','Semester5','Semester6','Semester7','Semester8']]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, Y_train)
model=GaussianNB()
model.fit(X_train, Y_train)

df7 = pd.read_csv('AdmitGrad1.csv')
df6=  pd.read_csv('graduate.csv')

st.set_page_config(page_title="My Page" ,layout="wide")
st.subheader("UNESWA GradeWise Tool")
mytext=""
err=-1
#get the id of the menu item clicked
with st.sidebar:
    selected=option_menu(
        menu_title:="menu",
        options:=["Home","Predict Grad","Data","Models","Contacts","CrossTab"],
        index:=0
        )
if selected=="Home":
        st.title("Home")
        st.write("According to the University of Eswatini website2, The University of Eswatini was established in 1972, with its specialisation in Agriculture. Today, it is the biggest and largest research universities in Eswatini. UNESWA has transformed into a dynamic university community of staff and students who come from a range of diverse backgrounds and cultures showcasing numerous global societies. The University was born from a vision to create a space for quality education and for new ideas to flourish. Over the course of its existence, UNESWA has been resilient in its commitment to academic quality. The main Clientele of the University is students as their core business is to offer Educational Services. Uneswa has 8 Faculties, 2 institutes and *10* departments. For the study we will focus on departments that deal directly with students (Registry and Faculties). Their processes start from application, registration, semester results and graduation. Academic Records are kept both by the Registry and Faculties. While registering Students, Assistant Registrar and Tutors are expected to advise students on courses to be taken. They use previous semester results to help and council student who are at risk of not graduating on time. More than 70% of the students who study at Uneswa are sponsored by government who only offers to pay for minimum required semester to graduate. Some of the students end up dropping out since they may not afford to pay the tuition fee for themselves.")
        c1,c2=st.columns(2)
        with c1:
            st.subheader("Table 1")
            st.dataframe(df6,hide_index=True)
        with c2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("Since Credit System, information from the University shows that 44.5% of Students do not Graduate on record time. There are a lot of factors that contribute to this This may include Demographic, Biographic, Programme of study, Assignments and Test and Social.")
        with c1:
            st.subheader("Table 2")
            st.dataframe(df7,hide_index=True)
        with c2:
            st.write("")
            st.write("")
            st.write("")
            st.write("Table 2 Shows that Female graduated on time when comparing them with males over the past year. Further study might have to look at why is this happening. For the study we have decided to include this feature on our Dataset")
        with c1:
            st.subheader("Table 3")
            st.dataframe(df8,hide_index=True)
        with c2:
            st.write("")
            st.write("")
            st.write("")
            st.write("Table 3 Shows were selected from the bachelors degree offered by uneswa. this programme all had 1 thing in common which is their minimum Graduation time is 8 Semesters(4 years). this programmes were used to create the dataset. This feature was included beacause of the graduation vs admitance ratio differ from programme to programme")
        
if selected=="Predict Grad":
    st.title("Predict Graduation and CGPA")
    with st.form(key="form1"):
        c1,c2,c3,c0=st.columns(4)
        c4,c5,c6,c7=st.columns(4)
        with c1:
            Sex=st.selectbox("Select Gender *",['Female','Male'])
        with c2:
            Age=st.slider(label ="Select Age Group *", min_value=16, max_value=60)
        with c3:
            Prog=st.selectbox("Select Programme of Study *",ProgrammeCode)
        with c0:
            Sponsored=st.selectbox("Sponsored *",['No','Yes'])
        with c4:
            Semester1=st.number_input("Enter Semester 1 GPA *",min_value=-1 ,max_value=6,value=-1,step=1)
        with c5:
            Semester2=st.number_input("Enter Semester 2 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c6:    
            Semester3=st.number_input("Enter Semester 3 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c7:    
            Semester4=st.number_input("Enter Semester 4 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c4:   
            Semester5=st.number_input("Enter Semester 5 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c5:   
            Semester6=st.number_input("Enter Semester 6 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c6:  
            Semester7=st.number_input("Enter Semester 7 GPA",min_value=-1,max_value=6,value=-1,step=1)
        with c7:  
            Semester8=st.number_input("Enter Semester 8 GPA",min_value=-1,max_value=6,value=-1,step=1)
            if(Sex=="Male"):
                Gender=1
            else:
                Gender=0
            if(Sponsored=="Yes"):
                Sponsor=1
            else:
                Sponsor=0   
                
            if (Age<=25):
                AgeGroup=0
            elif (Age<=35):
                AgeGroup=1
            else:
                AgeGroup=2
        with c4:
           
            if(Semester1!=-1):    
                if(Semester8!=-1):
                    grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3,Semester4,Semester5,Semester6,Semester7,Semester8])
                elif(Semester7!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3,Semester4,Semester5,Semester6,Semester7])   
                elif(Semester6!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3,Semester4,Semester5,Semester6])       
                elif(Semester5!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3,Semester4,Semester5])       
                elif(Semester4!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3,Semester4])       
                elif(Semester3!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2,Semester3])       
                elif(Semester2!=-1):
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1,Semester2])       
                else:
                     grad=predict([Gender,Sponsor,AgeGroup,int(Prog),Semester1]) 
                if(grad[0]==0):
                    err=0
                    if(grad[1]==0):        
                            mytext='Judging from your previous semesters, You are going to have chalenges in graduate on record time, you need to work a little bit harder to be able to graduate. With your current result you are at risk of getting a CGPA of '+str(grad[1])+" and not gradute. It is advisable you change programme"
                    if(grad[1]==1):        
                            mytext='Judging from your previous semesters, You are going to have chalenges in graduate on record time, you need to work a little bit harder to be able to graduate. With your current result you are at risk of getting a CGPA of '+str(grad[1])+". You will have a chalenge in completing the programme"
                    elif(grad[1]==2):
                            mytext="Judging from your previous semesters, You have a chance of not graduating on record time, you need to work a little bit harder to be able to archive a CGPA of better than "+str(grad[1])
                    elif(grad[1]>=3):
                            mytext="Judging from your previous semesters, You have a chance of not graduating on record time, you are on course of getting CGPA of better than "+str(grad[1])+" Keep up the good work!!!"
                            
                else:
                    err=1
                    if(grad[1]==2):
                            mytext="Judging from your previous semesters, You have a chance of graduating on record time, you need to work a little bit harder to be able to archive a CGPA of better than "+str(grad[1])
                    elif(grad[1]>=3):
                            mytext="Judging from your previous semesters, You have a chance of graduating on record time, you are on course of getting CGPA of better than "+str(grad[1])+" Keep up the good work!!!"
                            
            else:      
                err=-1
            submit =st.form_submit_button(label='Predicted')
if(err==1):
    st.success(mytext)
elif(err==0):
    st.error(mytext)
else:
    st.text("")           

if selected=="Data":
        st.title("About Data")
        st.text("Dataset")
        T=df[['Sex','Sponsored','AgeGroup','programme_code','Semester1','Semester2','Semester3',
      'Semester4','Semester5','Semester6','Semester7','Semester8','GRADUATED']]
        
        st.dataframe(T)
        plt.figure(figsize=(20, 16))
        corrmat = X.corr() # build the matrix of correlation from the dataframe using pandas.corr() function
        f, ax = plt.subplots(figsize=(12, 9)) # set up the matplotlib figure
        sns.heatmap(corrmat, vmax=1.0,cmap='Reds', vmin=-1.0, square=True, annot=True)
        st.pyplot(plt.gcf())
        st.text("Permutation Importance")
        # Calculate permutation importance
        results = permutation_importance(model, X_test, Y_test, scoring='accuracy')
        importance = results.importances_mean
        sorted_idx = importance.argsort()

        # Create a list of dictionaries for feature importance
        feature_importance_list = [{'Feature': X.columns[i], 'Score': importance[i]} for i in sorted_idx]

        
# Plot the permutation importance
        plt.figure(figsize=(10, 6))
        plt.barh([X.columns[i] for i in sorted_idx], importance[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.ylabel('Features')
        st.pyplot(plt.gcf())
        st.text("Bar chart of Graduate Data")
        unique_values,value_counts=np.unique(Y,return_counts=True)
        plt.bar(unique_values,value_counts, color='skyblue')
        plt.xlabel('Graduation')
        plt.ylabel('Frequency')
        plt.title('Bar chart of Graduate Data')
        st.pyplot(plt.gcf())
        st.text("Bar chart of Graduate Data")
        unique_values,value_counts=np.unique(Y_train,return_counts=True)
        plt.bar(unique_values,value_counts, color='red')
        plt.xlabel('Graduation')
        plt.ylabel('Frequency')
        plt.title('Bar chart of Traning Data 10% LESS')
        st.pyplot(plt.gcf())
        unique_values,value_counts=np.unique(y_train_resampled,return_counts=True)
        plt.bar(unique_values,value_counts, color='green')
        plt.xlabel('Graduation')
        plt.ylabel('Frequency')
        plt.title('Bar chart of Balanced Data')
        st.pyplot(plt.gcf())
        
if selected=="Models":
        st.title("Selected Model that were tested")
        with st.form(key="form1"):
            c1,c2=st.columns([9,1])
            with c1:
                Alg=st.selectbox("Select the Algithm to Check its perfomance",['Naive Bayes','Random Forest','SVM','GXBoost'])
                if Alg=="Naive Bayes":
                    model=GaussianNB()
                elif Alg=="Random Forest":
                    model=RandomForestClassifier()
                elif Alg=="SVM":
                    model=SVC(kernel='linear')
                elif Alg=="GXBoost":
                    model=xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
                    
            model.fit(X_train, Y_train)
            pred=model.predict(X_test)
            cm= confusion_matrix(Y_test,pred.round())
            sns.heatmap(cm,annot=True,fmt='g')
            plt.ylabel('Prediction', fontsize=13)
            plt.xlabel('Actual', fontsize=13)
            plt.title('Confussion Matrix', fontsize=17)
        
        with c2:
            submit =st.form_submit_button(label='View')
        precision = precision_score(Y_test,pred.round())
        recall = recall_score(Y_test,pred.round())
        accuracy=accuracy_score(Y_test,pred.round())
        f1=f1_score(Y_test,pred.round())
#y_pred_proba = regressor.predict_proba(X_test)[::,1]
#auc=metrics.roc_auc_score(Y_test,y_pred_proba)
        with c1:
            st.pyplot(plt.gcf())
        with c1:
            st.subheader("Precision:      "+str(precision))
            st.subheader('Recall     '+str(recall))
            st.subheader('Accuracy   '+str(accuracy))
            st.subheader('F1 score   '+str(f1))
        
                
                
if selected=="Contacts":
        st.title("Our Contacts")
        c1,c2,c3,c4=st.columns([1,2,1,2])
        with c1:
            st.image("me.jpg")
        with c2:
            data=[
                ['Name','Samkeliso Fakudze'],
                ['CellNumber','26876455633'],
                ['Phone Number','26825170141'],
                ['Email','ssfakudze@uniswa.sz']
                ]
            st.dataframe(data,hide_index=True)
        with c3:
            st.image("tgn.jpg")
        with c4:
            data=[
                ['Name','Thabani Ndzinisa'],
                ['CellNumber','26876415607'],
                ['Phone Number','26825170177'],
                ['Email','tgndzinisa@uniswa.sz']
                ]
            st.dataframe(data,hide_index=True)
if selected=="CrossTab":
            fig, ax = plt.subplots()
            cross_tab_prop =pd.crosstab(index=df['Sex'],
            columns=df['GRADUATED'],
            normalize="index")
            cross_tab =pd.crosstab(index=df['Sex'],
            columns=df['GRADUATED'])

            cross_tab.plot(kind='bar',
            stacked=True,
            colormap='tab10',
            figsize=(10, 6) ,ax=ax
                )
            ax.set_xticklabels(['Female','Male'])
            ax.legend( ['Yes','No'], title='OnTime Graduated', bbox_to_anchor=(1.0, 1), loc='upper left')
 
            plt.xlabel("Sex")
            plt.ylabel("Frequency")

            totals = cross_tab.sum(axis=1)

            i= 0
            j = 0
            for n, x in enumerate([*cross_tab.index.values]):
 
                for proportion in cross_tab_prop.loc[x]:
                    print("")
            plt.text(x=n,
            y=i + proportion*totals[j] *0.5,
            s=f'{np.round(proportion * 100, 1)}%',
            color="black",
            fontsize=12,
            fontweight="bold")
            i = i + proportion*totals[j]
            j= j +1

            i=0
            st.pyplot(plt.gcf())
            fig, ax = plt.subplots()
            cross_tab_prop =pd.crosstab(index=df['AgeGroup'],
            columns=df['GRADUATED'],
            normalize="index")
            cross_tab =pd.crosstab(index=df['AgeGroup'],
            columns=df['GRADUATED'])

            cross_tab.plot(kind='bar',
            stacked=True,
            colormap='tab10',
            figsize=(10, 6) ,ax=ax
                )
            ax.set_xticklabels(['less than 25','25-35','35 and above'])
            ax.legend( ['Yes','No'], title='OnTime Graduated', bbox_to_anchor=(1.0, 1), loc='upper left')
 
            plt.xlabel("Age Group")
            plt.ylabel("Frequency")

            totals = cross_tab.sum(axis=1)

            i= 0
            j = 0
            for n, x in enumerate([*cross_tab.index.values]):
 
                for proportion in cross_tab_prop.loc[x]:
                    print("")
            plt.text(x=n,
            y=i + proportion*totals[j] *0.5,
            s=f'{np.round(proportion * 100, 1)}%',
            color="black",
            fontsize=12,
            fontweight="bold")
            i = i + proportion*totals[j]
            j= j +1

            i=0
            st.pyplot(plt.gcf())
            fig, ax = plt.subplots()
            cross_tab_prop =pd.crosstab(index=df['Sponsored'],
            columns=df['GRADUATED'],
            normalize="index")
            cross_tab =pd.crosstab(index=df['Sponsored'],
            columns=df['GRADUATED'])

            cross_tab.plot(kind='bar',
            stacked=True,
            colormap='tab10',
            figsize=(10, 6) ,ax=ax
                )
            ax.set_xticklabels(['Self','Sponsored'])
            ax.legend( ['Yes','No'], title='OnTime Graduated', bbox_to_anchor=(1.0, 1), loc='upper left')
 
            plt.xlabel("Sponsor info")
            plt.ylabel("Frequency")

            totals = cross_tab.sum(axis=1)

            i= 0
            j = 0
            for n, x in enumerate([*cross_tab.index.values]):
 
                for proportion in cross_tab_prop.loc[x]:
                    print("")
            plt.text(x=n,
            y=i + proportion*totals[j] *0.5,
            s=f'{np.round(proportion * 100, 1)}%',
            color="black",
            fontsize=12,
            fontweight="bold")
            i = i + proportion*totals[j]
            j= j +1

            i=0
            st.pyplot(plt.gcf())
            

