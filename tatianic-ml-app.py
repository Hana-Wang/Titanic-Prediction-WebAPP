import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Titanic Prediction App

This app predicts the **Passenger Survive**!

""")

st.sidebar.header('User Input Features')

Pclass_List = [1, 2, 3]
Sex_List = ['male', 'female']
Age_List = [age for age in range(0, 101)]
#Age_List = [age for age in np.arange(0, 101.0, 0.5)]
SibSp_List = [sibsp for sibsp in range(0, 11)]
Parch_List = [parch for parch in range(0, 11)]
#Fare_List = [fare for fare in range(0, 600) ]
Embarked_List = ['S', 'C', 'Q']

# Collects user input features into dataframe
# Consider features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
def user_input_features():
    #Passenger Class
    Pclass = st.sidebar.selectbox('Select Passenger Class', Pclass_List)
    #Sex
    Sex = st.sidebar.selectbox('Select Passenger Gender', Sex_List)
    #Age
    Age = st.sidebar.selectbox('Select Passenger Age', Age_List)
    #Number of Siblings/Spouses Abroad
    SibSp = st.sidebar.selectbox("Selec Number of Siblings for Passenger", SibSp_List)
    #Number of Parents/Children Aboard
    Parch = st.sidebar.selectbox("Select Number of Parents/Children Aboard for Passenger", Parch_List)
    #Passenger Fare
    Fare = st.sidebar.slider('Passenger Fare is', 0.0, 600.0, 30.0)
    #Port of Embarkation
    Embarkation = st.sidebar.selectbox('Port of Embarkation', Embarked_List)

    data = {'pclass': Pclass,
            'sex': Sex,
            'age': Age,
            'sibsp': SibSp,
            'parch': Parch,
            'fare': Fare,
            'embarked': Embarkation }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Entered parameters')
st.write(df)



## Sex column; replace male with 0, female with 1
df.loc[df['sex'] == 'male', 'sex'] = 0
df.loc[df['sex'] == 'female', 'sex'] = 1

##change new data in column Embarked into numbers
df.loc[df['embarked'] == 'S', 'embarked'] = 0
df.loc[df['embarked'] == 'C', 'embarked'] = 1
df.loc[df['embarked'] == 'Q', 'embarked'] = 2




#titanic = pd.read_csv("titanic_train.csv")
titanic = pd.read_csv("titanic3.csv")

# Firstly, consider the possible data that make sense for
# predicing passenger survive,
# here, consider Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# clean dataset



## Age column;
# using the mean value to fill missing data in the Age
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())


## Fare column;
# using the mean value to fill missing data in the fare
titanic["fare"] = titanic["fare"].fillna(titanic["fare"].median())


## Sex column; replace male with 0, female with 1
titanic.loc[titanic['sex'] == 'male', 'sex'] = 0
titanic.loc[titanic['sex'] == 'female', 'sex'] = 1

## Embarked column;

## choose the most frequenct data in the Emarked column to fill missing data
titanic['embarked'] = titanic['embarked'].fillna('S')

##change data types into numbers
titanic.loc[titanic['embarked'] == 'S', 'embarked'] = 0
titanic.loc[titanic['embarked'] == 'C', 'embarked'] = 1
titanic.loc[titanic['embarked'] == 'Q', 'embarked'] = 2

# Consider features that make sense for predicting survive.
# Consider Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
x_data = titanic[predictors]
y_data = titanic['survived']

# Scale the predictors features data
#scaler = StandardScaler()
#x_data = scaler.fit_transform(x_data)
#df = scaler.fit_transform(df)
#Choose BaggingClassifier to train the model using

#Choose KNN to train the sampling datasets
#Choose the number of neighbors to use in KNN be 21
#knn = neighbors.KNeighborsClassifier(21)
#knn.fit(x_data, y_data)

#Choose BaggingClassifier
##bagging_knn = BaggingClassifier(knn, n_estimators = 20)
#train model using data
##bagging_knn.fit(x_data, y_data)


#prediction = bagging_knn.predict(df)
#prediction_proba = bagging_knn.predict_proba(df)

clf = RandomForestClassifier()
clf.fit(x_data, y_data)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Prediction')
survive_status = np.array(['Not Survived','Survived'])
st.write(survive_status[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
