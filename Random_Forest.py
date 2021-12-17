import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap

def random_forest(test,train):
    data_train= pd.read_csv(train)
    data_test=pd.read_csv(test)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    scalar_X=StandardScaler()
    X_Train=scalar_X.fit_transform(X_Train)
    X_Test=scalar_X.transform(X_Test)
    Y_Train=scalar_X.fit_transform(Y_Train)
    Y_Test=scalar_X.transform(Y_Test)

    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
    classifier.fit(X_Train,Y_Train)

    Y_pred = classifier.predict(X_Test)

    matriz_cunfusion=confusion_matrix(Y_Test,Y_pred)