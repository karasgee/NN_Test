import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2

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

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)

gp.fit(Feature, Output)

pred = gp.predict(Feature_V)

aver,std_dev,Mape_,r2_ = Scoring(Output_V,pred)
print(f'Mean : {aver}')
print(f'std  : {std_dev}')
print(f'r2   : {r2_}')
print("Optimized kernel:", gp.kernel_)
print("Constant (C):", gp.kernel_.k1.constant_value) 
print("RBF length scale:", gp.kernel_.k2.length_scale)
print(pred.reshape(-1,1))