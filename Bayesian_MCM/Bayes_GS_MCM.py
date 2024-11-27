import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV

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

# parameter GS
param_grid = {
    'alpha_1': [1e-11,1e-10,1e-9,1e-8,1e-7, 1e-6, 1e-5,1e-4],
    'alpha_2': [1e-11,1e-10,1e-9,1e-8,1e-7, 1e-6, 1e-5,1e-4],
    'lambda_1':[1e-11,1e-10,1e-9,1e-8,1e-7, 1e-6, 1e-5,1e-4],
    'lambda_2':[1e-11,1e-10,1e-9,1e-8,1e-7, 1e-6, 1e-5,1e-4],
    'tol' : [1e-11,1e-10,1e-9,1e-8,1e-7, 1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1]
}

grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=8,n_jobs=-1)
grid_search.fit(Feature, Output)
print("Best Parameters:", grid_search.best_params_)
pred = grid_search.predict(Vali_Feature)
# Model Scoring
Bay_r2 = r2(Vali_Output,pred)
Bay_mape = mape(Vali_Output,pred)
print(f'r2 : {Bay_r2}')
print(f'mape : {Bay_mape}')

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