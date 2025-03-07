import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# 定義生成器模型類別（與訓練時的模型結構相同）
class Generator(torch.nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# 載入模型與產生數據的函數
def generate_new_data(model_path, scaler_x_path, scaler_y_path, input_features):
    # 載入正規化器
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # 初始化生成器模型
    generator = Generator(input_dim=input_features.shape[1])
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # 正規化輸入特徵
    input_features_norm = scaler_x.transform(input_features)
    input_tensor = torch.tensor(input_features_norm, dtype=torch.float32)

    # 生成數據
    with torch.no_grad():
        generated_output = generator(input_tensor).numpy()

    # 反正規化輸出
    generated_output_original = scaler_y.inverse_transform(generated_output)
    return generated_output_original

if __name__ == "__main__":
    # 提供新的輸入特徵（替換為您的數據）
    new_features = np.random.uniform(1, 3, size=(20, 5))

    # 調用函數生成新數據
    result = generate_new_data(
        model_path="generator_model.pth",
        scaler_x_path="scaler_x.pkl",
        scaler_y_path="scaler_y.pkl",
        input_features=new_features
    )

    print("Generated Data:")
    print(result)
