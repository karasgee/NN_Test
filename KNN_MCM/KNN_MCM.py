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

# Define KNN
knn_regressor = KNeighborsRegressor(n_neighbors=5) # 使用 5 個最近鄰

# 訓練模型
knn_regressor.fit(Feature,Output)

# 預測
pred = knn_regressor.predict(Vali_Feature)

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