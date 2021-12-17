import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys

def regresion_logistica(train,test):
    data_train= pd.read_csv(train)
    data_test=pd.read_csv(test)

    #preparando datos

    #datos entrenamiento
    X_train = data_train[['24_7', 'alto_millaje', 'cambiar_llanta', 'carro_rentado', 'carro_viejo', 'conductor_experimentado', 'conductor_joven', 'dispuesto_pagar', 'historia_accidentes', 'maneja_mucho']]
    y_train = data_train['class']
    X_train = X_train.replace(['No'],'0')
    X_train = X_train.replace(['Si'],'1')
    y_train = y_train.replace(['plan_B'],'0')
    y_train = y_train.replace(['plan_C'],'1')

    #datos prueba
    X_test = data_test[['24_7', 'alto_millaje', 'cambiar_llanta', 'carro_rentado', 'carro_viejo', 'conductor_experimentado', 'conductor_joven', 'dispuesto_pagar', 'historia_accidentes', 'maneja_mucho']]
    y_test = data_test['class']
    X_test = X_test.replace(['No'],'0')
    X_test = X_test.replace(['Si'],'1')
    y_test = y_test.replace(['plan_B'],'0')
    y_test = y_test.replace(['plan_C'],'1')

    #regresion logistica
    regresion_logistica = LogisticRegression()
    x_nuevo = regresion_logistica.fit(X_train,y_train)
    y_pred = regresion_logistica.predict(X_test)

    #matriz de confusion 
    matriz_confusion=metrics.confusion_matrix(y_test,y_pred)
    print("Matriz de Confusion \n", matriz_confusion, "\n" )

    #metricas
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred, pos_label='1'))
    print("Precision:",metrics.precision_score(y_test, y_pred, pos_label='1'))
    print("F1-Score:",metrics.f1_score(y_test, y_pred, pos_label='1'))

    fig = go.Figure(data=[go.Table(header=dict(values=['24_7', 'alto_millaje', 'cambiar_llanta', 'carro_rentado', 'carro_viejo', 'conductor_experimentado', 'conductor_joven', 'dispuesto_pagar', 'historial_accidentes', 'maneja_mucho']),
    cells=dict(values=[x_nuevo.coef_[0][0], x_nuevo.coef_[0][1], x_nuevo.coef_[0][2], x_nuevo.coef_[0][3], x_nuevo.coef_[0][4], x_nuevo.coef_[0][5], x_nuevo.coef_[0][6], x_nuevo.coef_[0][7], x_nuevo.coef_[0][8], x_nuevo.coef_[0][9]]))])
    fig.show()


def main(train, test):
    regresion_logistica(train, test)


if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[-1]
    main(train, test)