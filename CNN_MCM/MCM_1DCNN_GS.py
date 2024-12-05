import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import r2_score  as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from keras.callbacks import EarlyStopping

# Load data
Training_set = np.loadtxt("SDS_487.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5]

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5]

# 正規化特徵
NM = MinMaxScaler()
NM.fit(Feature)
Feature = NM.transform(Feature)
Vali_Feature = NM.transform(Vali_Feature)

# 正規化輸出
Output = Output.reshape(-1, 1)
Vali_Output = Vali_Output.reshape(-1, 1)
NM_O = MinMaxScaler()
NM_O.fit(Output)
Output = NM_O.transform(Output)
Vali_Output = NM_O.transform(Vali_Output)

# 特徵轉換為 CNN 可接受的形狀
Feature = Feature.reshape(Feature.shape[0], Feature.shape[1], 1)
Vali_Feature = Vali_Feature.reshape(Vali_Feature.shape[0], Vali_Feature.shape[1], 1)

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(Feature, Output, test_size=0.2, random_state=42)

# 定義超參數空間
space = {
    'learning_rate': hp.choice('learning_rate', [ 0.001, 0.0001]),
    'batch_size': hp.choice('batch_size', [ 8, 16, ]),
    'kernel_size': hp.choice('kernel_size', [ 2, 3, 5, 7, 9]),
    'filters_layer1': hp.choice('filters_layer1', [8, 16, 32, ]),  # 第一層卷積的濾波器數
    'filters_layer2': hp.choice('filters_layer2', [8, 16, 32, ]), # 第二層卷積的濾波器數
    'activation': hp.choice('activation', ['relu', 'gelu']),
    'optimizer': hp.choice('optimizer', ['adam', 'nadam'])
}

# 模型訓練函數
def train_model(params):
    model = Sequential([
        Conv1D(filters=params['filters_layer1'], kernel_size=params['kernel_size'], activation=params['activation'], padding='same', input_shape=(Feature.shape[1], 1)),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=params['filters_layer2'], kernel_size=params['kernel_size'], activation=params['activation'], padding='same'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(16, activation=params['activation']),
        Dense(1, activation='linear')
    ])

    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    else:
        optimizer = Nadam(learning_rate=params['learning_rate'])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    early_stopping = EarlyStopping(monitor='val_mse', patience=30, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=10000, 
                        batch_size=params['batch_size'], 
                        callbacks=[early_stopping],
                        verbose=0)

    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

# 超參數調整
trials = Trials()
best = fmin(train_model, space, algo=tpe.suggest, max_evals=100, trials=trials)

# 最佳參數
print("Best hyperparameters:", best)

# 使用最佳參數重新訓練模型
optimal_params = {
    'learning_rate': [0.01, 0.001, 0.0001][best['learning_rate']],
    'batch_size': [1, 8, 16, 32, 64][best['batch_size']],
    'kernel_size': [1, 3, 5, 7, 9][best['kernel_size']],
    'filters_layer1': [8, 16, 32, 64, 128][best['filters_layer1']],
    'filters_layer2': [8, 16, 32, 64, 128][best['filters_layer2']],
    'activation': ['relu', 'gelu', 'swish','elu','sigmoid','linear'][best['activation']],
    'optimizer': ['adam', 'rmsprop', 'nadam'][best['optimizer']]
}

final_model = train_model(optimal_params)['model']
final_model.save('Optimized_CNN_MCM.keras')

# 評估最佳模型
test_loss, test_mse = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}")

# 預測
y_pred = final_model.predict(Vali_Feature)
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