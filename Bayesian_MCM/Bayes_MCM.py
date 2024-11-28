import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import BayesianRidge

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

# 正規化輸出
# NM_O = MinMaxScaler()
# NM_O.fit(Output)
# Output = NM_O.transform(Output).ravel()
# Vali_Output = NM_O.transform(Vali_Output).ravel()

# Define Model
# Best Parameters: {'alpha_1': 1e-11, 'alpha_2': 0.0001, 'lambda_1': 0.0001, 'lambda_2': 1e-11, 'tol': 1e-11}
model = BayesianRidge(
    tol = 1e-11,
    alpha_1=1e-11,
    alpha_2= 0.0001,
    lambda_1=0.0001,
    lambda_2=1e-11,
    compute_score=True,
    # fit_intercept=False
    )
model.fit(Feature,Output)

# Model Predict
pred = model.predict(Vali_Feature)
# pred = NM_O.inverse_transform(pred.reshape(-1, 1))

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

# 模型參數提取
print("Coef (Weights):", model.coef_)  # 回歸係數
print("Intercept:", model.intercept_)  # 截距
print("Alpha (Precision of weights):", model.alpha_)  # 權重的精度
print("Lambda (Precision of noise):", model.lambda_)  # 噪聲的精度
print("Sigma (Covariance of weights):", model.sigma_)  # 權重的協方差矩陣
