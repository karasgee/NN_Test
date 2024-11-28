import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
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

#定義模型
def build_model():
    model = Sequential()
    model.add(Dense(N1, input_dim=5, kernel_initializer='uniform', activation=af))
    model.add(Dense(N2, kernel_initializer='uniform', activation=af))
    model.add(Dense(N3, kernel_initializer='uniform', activation=af))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    return model

#Load data
Training_set = np.loadtxt("FOMCM_TrainSet.csv", delimiter=",")# "檔名.csv"
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# Hyperparameter
epoch = 10000
BS = 1 #Batch Size
LR = 0.0001 #Learning rate
N1 = 11 #Neural 1
N2 = 11  #Neural 2
N3 = 8 #Neural 3
af = 'gelu' #Activation Function
opt = 'nadam' #Opimizer
k = 9 # K-fold 的 k 值

# 訓練集 #此為五水準的情況下
Feature = Training_set[:,0:5] #前五排
Output = Training_set[:,5] #第五排
# 驗證集
Vali_Feature = Validation_set[:,0:5]
Vali_Output = Validation_set[:,5]
# 正規化數據
NM = MinMaxScaler() #定義縮放器
NM.fit(Feature) #定義縮放上下限
Feature  = NM.transform(Feature)
Vali_Feature = NM.transform(Vali_Feature)

# K-fold 以及 訓練模型
kf = KFold(n_splits=k, shuffle=True)
mse_scores = []
for train_index, val_index in kf.split(Feature):
    X_train, X_val = Feature [train_index], Feature [val_index]
    Y_train, Y_val = Output[train_index], Output[val_index]

    model = build_model()
    early_stopping = EarlyStopping(monitor='val_mse', patience=20, restore_best_weights=True)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epoch, batch_size=BS,callbacks=[early_stopping], verbose=1)
    mse_scores.append(history.history['val_mse'][-1])

#訓練評價
print("訓練中各fold的MSE:", mse_scores)
print("訓練中平均MSE:", np.mean(mse_scores))
print("訓練中標準差MSE:", np.std(mse_scores))

# 存檔
model.save('.keras') #'檔名.keras'

# 預測
Pred = model.predict(Vali_Feature)

# Scoring
aver,stan,mape_S,r2_S = Scoring (Vali_Output,Pred)
print(f'mean :{aver}')
print(f'std :{stan}')
print(f'R2 : {r2_S}')