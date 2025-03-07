import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from concurrent.futures import ThreadPoolExecutor
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
# 檢查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# -------------------------------
# 1. 讀取並預處理資料
# -------------------------------
Training_set = np.loadtxt("SDS_729.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集 (此為五水準情況)
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5].reshape(-1,1)

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5].reshape(-1,1)

# 正規化
NM = MinMaxScaler()
NM_O = MinMaxScaler()

NM.fit(Feature)
Feature = NM.transform(Feature)
Feature_V = NM.transform(Vali_Feature)

NM_O.fit(Output)
Output = NM_O.transform(Output)
Output_V = NM_O.transform(Vali_Output)

# -------------------------------
# 2. 定義損失函數
# -------------------------------
mse_loss = tf.keras.losses.MeanSquaredError()
def train_discriminator(real_data, fake_data, real_labels, fake_labels):
    for _ in range(3):
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    regression_loss = mse(real_data[:, -1], fake_data[:, -1])
    # d_loss_real, d_loss_fake 皆為 (loss, acc) tuple
    # 這邊取平均回傳
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0]+0.05*regression_loss)
    d_acc  = 0.5 * (d_loss_real[1] + d_loss_fake[1])
    return d_loss, d_acc
def combined_loss(y_true, y_pred):

    gan_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    regression_loss = mse_loss(y_true, y_pred)
    return gan_loss + 0.05 * regression_loss

# -------------------------------
# 3. 建立生成器與鑑別器
# -------------------------------
def build_generator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='relu')  # 回歸輸出
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 用 linear 輸出當作真(1)/假(0)的分數
    ])
    return model

generator = build_generator(input_dim=Feature.shape[1])
discriminator = build_discriminator(input_dim=Feature.shape[1] + 1)

# -------------------------------
# 4. 編譯模型
# -------------------------------
# 鑑別器先單獨編譯
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00000001)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
discriminator.trainable = False

# 建立整體GAN模型： z -> generator -> discriminator
z = layers.Input(shape=(Feature.shape[1],))
generated_y = generator(z)
d_input = layers.Concatenate()([z, generated_y])  # 輸入特徵 + 生成的 y
validity = discriminator(d_input)
gan = Model(z, validity)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
gan.compile(optimizer = generator_optimizer, loss=combined_loss)

# -------------------------------
# 5. 自定義訓練函式 
# -------------------------------



def train_generator(real_x):
    # 希望生成器生成的值能被 discriminator 判斷為真 (label=1)
    g_loss = gan.train_on_batch(real_x, tf.ones((real_x.shape[0], 1)))
    return g_loss

# -------------------------------
# 6. 訓練迴圈
# -------------------------------
epochs = 4000
batch_size = 512

for epoch in range(epochs):
    # 隨機抽 batch
    idx = np.random.randint(0, Feature.shape[0], batch_size)
    real_x = Feature[idx]
    real_y = Output[idx]

    # 生成假 y
    fake_y = generator(real_x, training=True)

    # 組合真實與假資料給鑑別器
    real_data = tf.concat([real_x, real_y], axis=1)
    fake_data = tf.concat([real_x, fake_y], axis=1)

    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    # 訓練鑑別器
    d_loss, d_acc = train_discriminator(real_data, fake_data, real_labels, fake_labels)
    # 訓練生成器
    g_loss = train_generator(real_x)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, D Acc: {d_acc}, G Loss: {g_loss[0]}")

# -------------------------------
# 7. 評估
# -------------------------------
predicted = generator.predict(Feature_V)
predicted = NM_O.inverse_transform(predicted)
actual = NM_O.inverse_transform(Output_V)

# 前10筆預測結果
for i in range(10):
    print(f"Actual: {actual[i][0]:.2f}, Predicted: {predicted[i][0]:.2f}")
