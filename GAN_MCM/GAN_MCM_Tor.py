import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape



def discriminator_loss(real_y, fake_y, d_real, d_fake, alpha):
    # 真實數據損失（真假判斷）
    bce_loss_real = bce_loss(d_real, torch.ones_like(d_real))
    # 假數據損失（真假判斷）
    bce_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
    # 回歸誤差損失
    mse_loss_reg = mse_loss(fake_y, real_y)
    # 總損失
    return bce_loss_real + bce_loss_fake + alpha * mse_loss_reg

# -------------------------------
# 1. 資料處理
# -------------------------------
# 讀取資料
Training_set = np.loadtxt("SDS_729.csv", delimiter=",")
Validation_set = np.loadtxt("FOMCM_Vali.csv", delimiter=",")

# 訓練集
Feature = Training_set[:, 0:5]
Output = Training_set[:, 5].reshape(-1, 1)

# 驗證集
Vali_Feature = Validation_set[:, 0:5]
Vali_Output = Validation_set[:, 5].reshape(-1, 1)

# 正規化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

Feature = scaler_x.fit_transform(Feature)
Output = scaler_y.fit_transform(Output)
Feature_V = scaler_x.transform(Vali_Feature)
Output_V = scaler_y.transform(Vali_Output)

# 轉為 PyTorch 張量
Feature = torch.tensor(Feature, dtype=torch.float32)
Output = torch.tensor(Output, dtype=torch.float32)
Feature_V = torch.tensor(Feature_V, dtype=torch.float32)
Output_V = torch.tensor(Output_V, dtype=torch.float32)

# -------------------------------
# 2. 建立模型
# -------------------------------
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.4),  # 添加 Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        input_data = torch.cat([x, y], dim=1)
        return self.model(input_data)


# 初始化模型
generator = Generator(input_dim=Feature.shape[1])
discriminator = Discriminator(input_dim=Feature.shape[1])

# -------------------------------
# 3. 設定優化器與損失函數
# -------------------------------

g_optimizer = optim.Adam(generator.parameters(), lr=0.00005)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
lr_scheduler_d = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=5000, gamma=0.9)
lr_scheduler_g = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=5000, gamma=0.9)

# 損失函數
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

# -------------------------------
# 4. 訓練迴圈
# -------------------------------
epochs = 50000
batch_size = 256


for epoch in range(epochs):
    # 隨機抽 batch
    idx = np.random.randint(0, Feature.shape[0], batch_size)
    real_x = Feature[idx]
    real_y = Output[idx]

    # 生成假 y
    fake_y = generator(real_x).detach()

    # 訓練鑑別器
    real_labels = torch.ones((batch_size, 1))
    fake_labels = torch.zeros((batch_size, 1))

    d_real = discriminator(real_x, real_y)
    d_fake = discriminator(real_x, fake_y)

    d_loss = discriminator_loss(real_y, fake_y, d_real, d_fake, alpha=0.2)

    # 計算正確率
    real_acc = torch.mean((d_real >= 0.5).float())  # 真實數據的正確率
    fake_acc = torch.mean((d_fake < 0.5).float())   # 假數據的正確率
    d_accuracy = (real_acc + fake_acc) / 2          # 平均正確率

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # 訓練生成器
    fake_y = generator(real_x)
    g_loss_gan = bce_loss(discriminator(real_x, fake_y), real_labels)
    g_loss_reg = mse_loss(fake_y, real_y)
    g_loss = g_loss_gan + 0.5 * g_loss_reg

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # 訓練過程輸出
    if epoch % 1000 == 0:
        print(f"Epoch {epoch},D acc {d_accuracy}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    if epoch % 10 == 0:
       with torch.no_grad():
        predicted = generator(Feature_V).numpy()
        actual = Output_V.numpy()
        predicted = scaler_y.inverse_transform(predicted)
        actual = scaler_y.inverse_transform(actual)
        r2_ = r2_score(actual,predicted)
        if r2_ > 0.98:
            print(f'Break in : {epoch}')
            torch.save(generator.state_dict(), "generator_model.pth")
            break

# -------------------------------
# 5. 評估
# -------------------------------
with torch.no_grad():
    predicted = generator(Feature_V).numpy()
    actual = Output_V.numpy()
    predicted = scaler_y.inverse_transform(predicted)
    actual = scaler_y.inverse_transform(actual)

print(mape(actual,predicted))
# 打印前 10 筆預測結果
for i in range(10):
    print(f"Actual: {actual[i][0]:.2f}, Predicted: {predicted[i][0]:.2f}")


import joblib
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

