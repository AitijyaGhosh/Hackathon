import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


health_data = pd.read_csv('E:\Computer\python\Algo_Avengers\diabetes\diabetes_012_health_indicators_BRFSS2015.csv')
X = health_data.drop(columns = ['Diabetes_012', 'Sex', 'Income', 'Education', 'MentHlth', 'NoDocbcCost', 'AnyHealthcare', 'Fruits', 'Veggies', 'CholCheck'])
y = health_data["Diabetes_012"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def NBmethod():

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    

    y_predict= nb_model.predict(X_test)

    return y_predict

def report():
    print(classification_report(y_test, y_predict))

y_predict = NBmethod()
report()
