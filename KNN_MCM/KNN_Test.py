import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler



# 加載加州房價數據
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立 KNN 模型
knn = KNeighborsRegressor(n_neighbors=3)  # k 值可以調整
knn.fit(X_train, y_train)

# 預測與評估
y_pred = knn.predict(X_test)
r2_S = r2(y_test, y_pred)
print(f"Mean Squared Error: {r2_S}")


