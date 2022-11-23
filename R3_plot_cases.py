# Bibliotecas de Machine Learning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

#Bibliotecas de para leitura e tratamento de dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Leitura do arquivo Breast Cancer Prediction.csv
project_data = pd.read_csv("E:\Acadêmico\Python\Machine Learning\Breast Cancer Predication\Breast Cancer Prediction.csv", sep = ';')
project_data = project_data.drop(columns=["id"]) #Remoção de dados de cadastro dos pacientes

#Tradução do Cabeçalho
project_data.rename(columns = {"diagnosis":"Diagnóstico",
                               "Radius_mean":"Raio Médio",                          
                               "Texture_mean":"Textura Média",                      
                               "perimeter_mean":"Perímetro Médio",                  
                               "area_mean":"Área Média",                            
                               "smoothness_mean":"Suavidade Média",                 
                               "compactness_mean":"Compacidade Média",              
                               "concavity_mean":"Concavidade Média",                
                               "concave points_mean":"Pontos Côncavos Médios",      
                               "symmetry_mean":"Simetria Média",                    
                               "fractal_dimension_mean":"Dimensão Fractal Média",

                               "radius_se":"Erro Padrão do Raio",                          
                               "texture_se":"Erro Padrão da Textura",                      
                               "perimeter_se":"Erro Padrão do Perímetro",                  
                               "area_se":"Erro Padrão da Área",                            
                               "smoothness_se":"Erro Padrão da Suavidade",                 
                               "compactness_se":"Erro Padrão da Compacidade",              
                               "concavity_se":"Erro Padrão da Concavidade",                
                               "concave points_se":"Erro Padrão dos Pontos Côncavos",      
                               "symmetry_se":"Erro Padrão da Simetria",                    
                               "fractal_dimension_se":"Erro Padrão da Dimensão Fractal",
                               
                               "radius_worst":"Pior Raio",
                               "texture_worst":"Pior Textura",
                               "perimeter_worst":"Pior Perímetro",
                               "area_worst":"Pior Área",
                               "smoothness_worst":"Pior Suavidade",
                               "compactness_worst":"Pior Compacidade",
                               "concavity_worst":"Pior Concavidade",
                               "concave points_worst":"Piores Pontos Côncavos",
                               "symmetry_worst":"Pior Simetria",
                               "fractal_dimension_worst":"Pior Dimensão Fractal"}, inplace = True)

#Identificando os dados correspondentes de cada classe
dM = (project_data.loc[project_data['Diagnóstico'] == "M"]).values
dB = (project_data.loc[project_data['Diagnóstico'] == "B"]).values

#Criação dos Gráficos de Ocorrência das Característica Médias dos Nódulo
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(dB[:,2], dB[:,3],dB[:,1], marker="o", color = "r")
ax.scatter(dM[:,2], dM[:,3],dM[:,1], marker="o", color = "b")
ax.set_xlabel('Textura Média')
ax.set_ylabel('Perímetro Médio')
ax.set_zlabel('Raio Médio')
plt.legend(["Maligno","Benigno"])
plt.title("Distribuição dos Casos no R3")
plt.show()