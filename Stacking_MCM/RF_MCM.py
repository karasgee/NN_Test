import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error  as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
import json
import pickle
from sklearn.model_selection import GridSearchCV
import sys

file_path = "F:\\NN_Test\\Stacking_MCM\\"
# load data
dataset = np.loadtxt(file_path+"FOMCM_TrainSet.csv",delimiter = ",")
dataset_V = np.loadtxt(file_path+"FOMCM_Vali.csv",delimiter = ",")

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
# paraset
param_grid = {
    'n_estimators': [50,70,100,110,140,150,200,400,550,500,1000],
    'max_depth': [None],
    'max_features': ['sqrt','log2',2,10],
    # 'bootstrap': [True,False],
    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    # 'oob_score':[True,False],
    # 'min_impurity_decrease':[0.0,1.0],
    'max_samples':[None,100,200,400],
}

# Training Model(RF)
RF_model = RandomForestRegressor(random_state=42,verbose=1)
grid_search = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=10, scoring='r2', n_jobs=-1)
grid_search.fit(Feature,Output)
print("最佳參數組合：", grid_search.best_params_)
print("最佳模型的性能：", grid_search.best_score_)

# 預測
Pred = grid_search.predict(Feature_V )

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
print(f'R2 : {r2_S}')

# 存模型
with open('MCM_RF.pkl', 'wb') as file:
    pickle.dump(grid_search, file)
print("存檔 : 'MCM_RF.pkl'")

param = grid_search.best_params_
param['mape'] = mape_S
param['R2'] = r2_S
param['mse mean'] = aver
param['mse std'] =stan[0]

with open("RF_MCM.json", "w", encoding="utf-8") as file:
    json.dump(param, file, indent=4)  