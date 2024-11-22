import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

#Load data
Training_set = np.loadtxt("FOMCM_TrainSet.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:,0:5]
Output = Training_set[:,5]

# 驗證集
Vali_Feature = Validation_set[:,0:5]
Vali_Output = Validation_set[:,5]

# 正規化特徵
NM = MinMaxScaler()
NM.fit(Feature)
Feature  = NM.transform(Feature)
Vali_Feature = NM.transform(Vali_Feature)

# 正規化輸出
Output = Output.reshape(-1,1)
Vali_Output = Vali_Output.reshape(-1,1)
NM_O = MinMaxScaler()
NM_O.fit(Output)
Output = NM_O.transform(Output)
Vali_Output = NM_O.transform(Vali_Output)

# 特徵轉換為1N CNN 可接受的形狀
Feature = Feature.reshape(Feature.shape[0], Feature.shape[1], 1) #Tr
Vali_Feature = Vali_Feature.reshape(Vali_Feature.shape[0], Vali_Feature.shape[1], 1) #Va

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(Feature, Output, test_size=0.2, random_state=42)

# 建立一維 CNN 模型
model = Sequential([
    Conv1D(8, kernel_size=1, activation='relu', input_shape=(Feature.shape[1], 1)),
    MaxPooling1D(pool_size=1),
    Conv1D(8, kernel_size=1, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')  # 單輸出回歸
])

# 編譯模型
model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mse'])

# 訓練模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# 評估模型
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# 預測
y_pred = model.predict(Vali_Feature)
y_pred = NM_O.inverse_transform(y_pred)  # 還原標準化的結果
CNN_Mape = mape(Vali_Output,y_pred)
CNN_r2 = r2(Vali_Output,y_pred)

print(f'CNN_Mape : {CNN_Mape}')
print(f'CNN_R2 : {CNN_r2}')

def stdevper(Ground_Truth,Predict):
    Ground_Truth=Ground_Truth.reshape(-1,1)
    Predict = Predict.reshape(-1,1)
    diff = np.abs(Ground_Truth-Predict)/Ground_Truth
    std_dev = np.std(diff.reshape(-1,1),axis=0)
    aver = np.mean(diff.reshape(-1,1))
    return aver,std_dev

aver,stan =stdevper(Vali_Output,y_pred)
print(f'mean :{aver}')
print(f'std :{stan}')
print(y_pred)

# 存檔
model.save('CNN_MCM.keras')