import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

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
Training_set = np.loadtxt("", delimiter=",")# "檔名.csv"
Validation_set = np.loadtxt("", delimiter=",")

# Hyperparameter
epoch = 10000
BS = 1 #Batch Size
LR = 0.005 #Learning rate
N1 = 5 #Neural 1
N2 = 5 #Neural 2
N3 = 5 #Neural 3
af = 'relu' #Activation Function
opt = 'Adam' #Opimizer
k = 9 # K-fold 的 k 值

# 訓練集 #此為五水準的情況下
Feature = Training_set[:,0:5] #前五排
Output = Training_set[:,5] #第五排
# 驗證集
Vali_Feature = Validation_set[:,0:5]
Vali_Output = Validation_set[:,5]
# 正規化數據
NM = MinMaxScaler()#定義縮放器
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
    early_stopping = EarlyStopping(monitor='val_mse', patience=30, restore_best_weights=True)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epoch, batch_size=BS,callbacks=[early_stopping], verbose=1)
    mse_scores.append(history.history['val_mse'][-1])

#訓練評價
print("訓練中各fold的MSE:", mse_scores)
print("訓練中平均MSE:", np.mean(mse_scores))
print("訓練中標準差MSE:", np.std(mse_scores))

# 存檔
model.save('.keras') #'檔名.keras'