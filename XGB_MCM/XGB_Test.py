import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

# Load data
Training_set = np.loadtxt("FOMCM_TrainSet.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5].reshape(-1, 1)

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5].reshape(-1, 1)

# 正規化特徵
NM_F = MinMaxScaler()
NM_F.fit(Feature)
Feature = NM_F.transform(Feature)
Vali_Feature = NM_F.transform(Vali_Feature)

# 正規化輸出
NM_O = MinMaxScaler()
NM_O.fit(Output)
Output = NM_O.transform(Output)
Vali_Output = NM_O.transform(Vali_Output)

# 轉換XGB格式 (DMatrix)
Train_data = xgb.DMatrix(Feature, label=Output)
Vali_data = xgb.DMatrix(Vali_Feature, label=Vali_Output)

# paraset
params = {
    "booster": "gblinear",
    "nthread": -1,
    "eta": 0.1,
    # "max_depth": 6,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    # "verbosity" :  3,
    # "subsample" : 0.95 
}

# XGB Model Training
model = xgb.train(
    params,
    Train_data,
    num_boost_round=1000,
    # evals=[(Vali_data, "test")],
    # early_stopping_rounds= 1000,
)

# XGB pred
pred = model.predict(Vali_data).reshape(-1, 1)

# 逆變換預測值
pred_original = NM_O.inverse_transform(pred)
Vali_Output_original = NM_O.inverse_transform(Vali_Output)

# XGB Scoring
XGB_r2 = r2(Vali_Output_original, pred_original)
XGB_mape = mape(Vali_Output_original, pred_original)
print(f'r2 : {XGB_r2}')
print(f'mape : {XGB_mape}')

def stdevper(Ground_Truth, Predict):
    diff = np.abs(Ground_Truth - Predict) / Ground_Truth
    std_dev = np.std(diff, axis=0)
    aver = np.mean(diff)
    return aver, std_dev

aver, stan = stdevper(Vali_Output_original, pred_original)
print(f'mean mse :{aver}')
print(f'std mse :{stan}')
