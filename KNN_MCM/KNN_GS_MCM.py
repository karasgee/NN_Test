from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
import numpy as np

# Load data
Training_set = np.loadtxt("FOMCM_TrainSet.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5]

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5]

# 正規化特徵
NM_F = MinMaxScaler()
NM_F.fit(Feature)
Feature = NM_F.transform(Feature)
Vali_Feature = NM_F.transform(Vali_Feature)

# Parameter Set
params = {
    'n_neighbors': range(1, 81),
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3] ,
    'algorithm': ['auto'],
}

# Define model
knn_regressor = KNeighborsRegressor()

# model training
grid_search = GridSearchCV(knn_regressor, params, cv=9,n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(Feature,Output)

# Model Predict
# best_k = grid_search.best_params_['n_neighbors']
# print(f"最佳的 k 值: {best_k}")
print(f"最佳參數: {grid_search.best_params_}")
pred = grid_search.predict(Vali_Feature)

# Model Scoring
KNN_r2 = r2(Vali_Output,pred)
KNN_mape = mape(Vali_Output,pred)
print(f'r2 : {KNN_r2}')
print(f'mape : {KNN_mape}')

def stdevper(Ground_Truth,Predict):
    Ground_Truth=Ground_Truth.reshape(-1,1)
    Predict = Predict.reshape(-1,1)
    diff = np.abs(Ground_Truth-Predict)/Ground_Truth
    std_dev = np.std(diff.reshape(-1,1),axis=0)
    aver = np.mean(diff.reshape(-1,1))
    return aver,std_dev

aver,stan = stdevper(Vali_Output,pred)
print(f'mean mse :{aver}')
print(f'std mse :{stan}')
print(pred.reshape(-1,1))
