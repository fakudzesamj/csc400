import pandas as pd
import numpy as np
def predict(input_data):
    df = pd.read_csv('csc499dataset.csv')
    if (len(input_data))==11:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3',
      'Semester4','Semester5','Semester6','Semester7','Semester8']]
    if  (len(input_data))==10:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3',
              'Semester4','Semester5','Semester6','Semester7']]
    if  (len(input_data))==9:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3',
              'Semester4','Semester5','Semester6']]
    if  (len(input_data))==8:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3',
              'Semester4','Semester5']]
    if  (len(input_data))==7:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3',
              'Semester4']]
    if  (len(input_data))==6:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2','Semester3']]
    if  (len(input_data))==5:
        X=df[['Sex','AgeGroup','programme_code','Semester1','Semester2']]
    if  (len(input_data))==4:
        X=df[['Sex','AgeGroup','programme_code','Semester1']]
    Y=df["GRADUATED"]
    Z=df["CGPA"]
    XX=X.values
    YY=Y.values
    ZZ=Z.values
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(XX,YY,test_size=0.1,random_state=42)
    from sklearn.model_selection import train_test_split
    W_train,W_test,Z_train,Z_test=train_test_split(XX,ZZ,test_size=0.1,random_state=42)
    from sklearn.naive_bayes import GaussianNB
    modelGrad=GaussianNB()
    modelCGPA=GaussianNB()
    modelGrad.fit(X_train, Y_train)
    modelCGPA.fit(W_train, Z_train)
    predGrad = modelGrad.predict([input_data])
    predCGPA = modelCGPA.predict([input_data])
    return np.array([predGrad,predCGPA])
   