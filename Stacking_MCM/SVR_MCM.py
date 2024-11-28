import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score  as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR
import pickle
from sklearn.model_selection import GridSearchCV
import json

# load data
dataset = np.loadtxt("FOMCM_TrainSet.csv",delimiter = ",")
dataset_V = np.loadtxt("FOMCM_Vali.csv",delimiter = ",")

# Faeture
Feature = dataset[:,:5]
Feature_V = dataset_V[:,:5]

# Output 
Output = dataset[:,5]
Output_V = dataset_V[:,5]

# Normalize
NM = MinMaxScaler()
NM.fit(Feature)
Feature = NM.transform(Feature)
Feature_V = NM.transform(Feature_V)

param_grid = [
    # 線性核
    # {
    #     'kernel': ['linear'],
    #     'C': [0.1, 0.5, 0.81, 3, 510],
    #     'epsilon': [0.001, 0.01, 0.1, 1, 10],
    #     'shrinking': [True, False],
    #     'cache_size': [100, 200, 300]
    # },
    # # 多項式核
    # {
    #     'kernel': ['poly'],
    #     'C': [0.1, 0.5, 0.81, 3, 510],
    #     'epsilon': [0.001, 0.01, 0.1, 1, 10],
    #     'degree': [2, 3, 4],
    #     'gamma': ['scale', 'auto', 2],
    #     'shrinking': [True, False],
    #     'cache_size': [100, 200, 300]
    # },
    # RBF 核
    {
        'kernel': ['rbf'],
        'C': [0.1, 0.5, 0.81, 3],
        # 'epsilon': [10],
        'gamma': [0.001, 0.01, 0.1, 1, 10],
        # 'shrinking': [True, False],
        'cache_size': [100, 200, 300]
    }
    ,
    # Sigmoid 核
    # {
    #     'kernel': ['sigmoid'],
    #     'C': [0.1, 0.5, 0.81, 3, 510],
    #     'epsilon': [0.001, 0.01, 0.1, 1, 10],
    #     'gamma': ['scale', 'auto', 2],
    #     'shrinking': [True, False],
    #     'cache_size': [100, 200, 300]
    # }
]

# Training Model
svr_model = SVR(verbose=2)
grid_search_SVR = GridSearchCV(svr_model, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_SVR.fit(Feature, Output)
print("最佳參數組合：", grid_search_SVR.best_params_)
# print("最佳模型的性能：", grid_search_SVR.best_score_)

# Predict
Pred = grid_search_SVR.predict(Feature_V)

# Scoring Function
def Scoring(Ground_Truth,Predict):
    # 定義 Ground_Truth 跟 Predict 值
    Ground_Truth=Ground_Truth.reshape(-1,1)
    Predict = Predict.reshape(-1,1)
    # 定義 差異值 diff 與標準差計算式
    diff = np.abs(Ground_Truth-Predict)/Ground_Truth
    std_dev = np.std(diff.reshape(-1,1),axis=0)
    aver = np.mean(diff.reshape(-1,1))
    #定義 mape 與 r2
    Mape_ = mape(Ground_Truth,Predict)
    r2_ = r2(Ground_Truth,Predict)
    # 回傳值
    return aver,std_dev,Mape_,r2_

aver,stan,mape_S,r2_S = Scoring (Output_V,Pred)
print(f'mean :{aver}')
print(f'std :{stan}')
# print(f'mape: {mape_S}')
print(f'R2 : {r2_S}')

# 存模型
with open('MCM_SVR.pkl', 'wb') as file:
    pickle.dump(grid_search_SVR, file)
print("存檔 : 'MCM_SVR.pkl ^ SVR_MCM.txt'")

# 存参數 ()json 
param = grid_search_SVR.best_params_
param['mape'] = mape_S
param['R2'] = r2_S
param['mse mean'] = aver
param['mse std'] =stan[0]

with open("SVR_MCM.json", "w", encoding="utf-8") as file:
    json.dump(param, file, indent=4)  


