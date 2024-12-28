import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2

# Load data
Training_set = np.loadtxt("FOMCM_TrainSet.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5].reshape(-1,1)

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5].reshape(-1,1)

# 正規化特徵
NM_F = MinMaxScaler()
NM_F.fit(Feature)
Feature = NM_F.transform(Feature)
Vali_Feature = NM_F.transform(Vali_Feature)

# 正規化輸出
# NM_O = MinMaxScaler()
# NM_O.fit(Output)
# Output = NM_O.transform(Output)
# Vali_Output = NM_O.transform(Vali_Output)

# 轉換XGB格式 (DMatrix)
Train_data = xgb.DMatrix(Feature,label = Output)
Vali_data = xgb.DMatrix(Vali_Feature,label = Vali_Output)

# paraset
params = {
    # 通用參數
    "booster": "gbtree",            # 模型類型，可選 "gbtree", "gblinear", "dart"
    "nthread": -1,                  # 使用的線程數，-1 代表自動檢測

    # 樹模型參數
    "eta": 0.0001,                    # 學習率，範圍 (0, 1]
    "max_depth": 100,              # 樹的最大深度，範圍 [1, ∞)
    # "tree_method": "hist",          # 樹構建算法，可選 "auto", "exact", "approx", "hist", "gpu_hist"

    # 學習目標相關參數
    "objective": "reg:squarederror", # 學習目標，可選 "reg:squarederror", "binary:logistic", "multi:softmax", 等
    "eval_metric": "rmse",          # 評估指標，可選 "rmse", "logloss", "error", "auc", 等

}

# XGB Model Training
model = xgb.train(
    params,
    Train_data,
    num_boost_round = 100000,
    evals = [(Vali_data,"test")],
    early_stopping_rounds= 20 ,
    )

# XGB pred
pred = model.predict(Vali_data).reshape(-1,1)
# pred = NM_O.inverse_transform(pred)

# XGB Scoring
XGB_r2 = r2(Vali_Output,pred)
XGB_mape = mape(Vali_Output,pred)
print(f'r2 : {XGB_r2}')
print(f'mape : {XGB_mape}')

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
# print(pred)

