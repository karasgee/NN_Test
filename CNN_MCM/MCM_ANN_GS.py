import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

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
    'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001]),
    'batch_size': hp.choice('batch_size', [8, 16, 32, 64]),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7]),
    'activation': hp.choice('activation', ['relu', 'gelu', 'swish']),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'nadam'])
}

# 模型訓練函數
def train_model(params):
    model = Sequential([
        Conv1D(8, kernel_size=params['kernel_size'], activation=params['activation'], input_shape=(Feature.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(16, kernel_size=params['kernel_size'], activation=params['activation']),
        MaxPooling1D(pool_size=2),
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

    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=50, 
                        batch_size=params['batch_size'], 
                        verbose=0)

    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

# 超參數調整
trials = Trials()
best = fmin(train_model, space, algo=tpe.suggest, max_evals=50, trials=trials)

# 最佳參數
print("Best hyperparameters:", best)

# 使用最佳參數重新訓練模型
optimal_params = {
    'learning_rate': [0.01, 0.001, 0.0001][best['learning_rate']],
    'batch_size': [8, 16, 32, 64][best['batch_size']],
    'kernel_size': [3, 5, 7][best['kernel_size']],
    'activation': ['relu', 'tanh', 'swish'][best['activation']],
    'optimizer': ['adam', 'rmsprop', 'nadam'][best['optimizer']]
}

final_model = train_model(optimal_params)['model']
final_model.save('Optimized_CNN_MCM.keras')

# 評估最佳模型
test_loss, test_mse = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}")
