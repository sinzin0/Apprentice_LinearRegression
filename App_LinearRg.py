# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:28:35 2024

@author: jyshin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
   

def TrainData(data):
    
    X = data[['시간', '일', '월',  '일조(hr)', '일사(MJ/m2)', '지면온도(°C)', '습도(%)', '풍속(m/s)', '기온(°C)' ]]
    y = data['시정(10m)']    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #pca = PCA(n_components=8)
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)

    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)    

    # 각 특성에 대한 회귀 계수
    coefficients = model.coef_
    
    # 각 특성의 중요도 출력
    for feature, coef in zip(X_train.columns, coefficients):
        print(f'{feature}: {coef:.4f}')
    
    # 예측값 생성
    y_pred = model.predict(X_test_scaled)
    
    # 결과 시각화
    plt.figure(figsize=(8, 6))
    
    # 실제 값 vs 예측 값 비교
    plt.scatter(y_test, y_pred, color='b', label="Predicted vs Original", linewidth=1)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label="Good Predicted")
    
    # 그래프 제목 및 레이블
    plt.title("Prediction Result( Original vs Predicted )")
    plt.xlabel("Original")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()    
    
    
    # 1. R² (결정 계수)
    r2 = r2_score(y_test, y_pred)
    
    # 2. 평균 제곱 오차 (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # 3. 평균 절대 오차 (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 결과 출력
    print(f"결정 계수 (R²): {r2:.4f}")
    print(f"평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"평균 절대 오차 (MAE): {mae:.4f}")
    
    

# 데이터 생성
#DataMerge()

# 데이터 읽기
data = pd.read_csv("merged_data.csv", encoding='utf-8')

# 데이터 정규화 
#_processData = Standardization(data)

# 데이터 학습
TrainData(data)