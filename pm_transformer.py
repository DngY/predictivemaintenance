import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

# 生成数据（保持不变）
def generate_data(filename="train.csv"):
    num_records = 1000
    start_time = pd.Timestamp("2024-03-09 10:52:00")
    interval = pd.Timedelta(minutes=5)
    timestamps = [start_time + i * interval for i in range(num_records)]
    np.random.seed(42)
    temperature = np.random.normal(loc=0.06, scale=0.002, size=num_records)
    vibration = np.random.normal(loc=0.075, scale=0.002, size=num_records)
    motor_rpm = np.random.normal(loc=0.045, scale=0.001, size=num_records)
    motor_amps = np.random.normal(loc=0.084, scale=0.002, size=num_records)
    anomaly_start_index = num_records - 200
    temperature[anomaly_start_index:] += np.random.uniform(0.005, 0.01, size=200)
    vibration[anomaly_start_index:] += np.random.uniform(0.003, 0.007, size=200)
    motor_rpm[anomaly_start_index:] -= np.random.uniform(0.002, 0.005, size=200)
    motor_amps[anomaly_start_index:] += np.random.uniform(0.005, 0.01, size=200)
    anomaly_flag = np.zeros(num_records, dtype=int)
    anomaly_flag[anomaly_start_index:] = 1
    data = {
        "TEMPERATURE": temperature, "VIBRATION": vibration,
        "MOTOR_RPM": motor_rpm, "MOTOR_AMPS": motor_amps,
        "MEASURE_TS": timestamps, "ANOMALY_FLAG": anomaly_flag
    }
    data_pd = pd.DataFrame(data)
    data_pd.to_csv(filename, index=False)

# Transformer 自编码器（增强容量）
class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=8, num_layers=3):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        return x

# 参数
timesteps = 1  # 保持单时间步
n_features = 4

def add_noise(data, noise_factor=0.01):
    """
    为数据添加高斯噪声
    :param data: 输入数据，形状 (n_samples, n_features)
    :param noise_factor: 噪声幅度，相对于数据标准差的比例
    :return: 加噪后的数据
    """
    noise = np.random.normal(loc=0, scale=noise_factor * np.std(data, axis=0), size=data.shape)
    return data + noise
def train():
    print("......开始模型训练......")
    data_pd = pd.read_csv("train.csv")
    # 只用正常数据训练（前 800 条）
    X = data_pd[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].iloc[:800].to_numpy()
    # 添加噪声
    X_noisy = add_noise(X, noise_factor=0.01)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_noisy)
    # 保存 scaler
    with open("./models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    n_samples = X_train.shape[0] // timesteps  # 800
    X_train = X_train[:n_samples * timesteps].reshape(n_samples, timesteps, n_features)
    X_train = torch.FloatTensor(X_train)
    print("X_train 形状:", X_train.shape)  # (800, 1, 4)

    model = TransformerAutoencoder(n_features=n_features, d_model=64, nhead=8, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200  # 增加训练轮次
    batch_size = 32  # 调整 batch_size
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, n_samples, batch_size):
            batch = X_train[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / (n_samples // batch_size):.6f}")

    torch.save(model.state_dict(), "./models/transformer_autoencoder.pth")
    print("模型已保存到 ./models/transformer_autoencoder.pth")

def predict():
    model = TransformerAutoencoder(n_features=n_features, d_model=64, nhead=8, num_layers=3)
    model.load_state_dict(torch.load("./models/transformer_autoencoder.pth"))
    model.eval()
    
    # 加载 scaler
    with open("./models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 读取完整训练数据计算阈值
    data_pd = pd.read_csv("train.csv")
    X_train_full = data_pd[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].to_numpy()
    X_train_full = scaler.transform(X_train_full)  # 使用训练时的 scaler
    n_samples = X_train_full.shape[0] // timesteps  # 1000
    X_train = X_train_full[:n_samples * timesteps].reshape(n_samples, timesteps, n_features)
    X_train = torch.FloatTensor(X_train)
    
    with torch.no_grad():
        X_train_pred = model(X_train)
        mse_train = torch.mean((X_train - X_train_pred) ** 2, dim=(1, 2)).numpy()
        threshold = np.mean(mse_train) + 3 * np.std(mse_train)  # 均值 + 3 倍标准差
        print("阈值:", threshold)

    # 新数据
    new_data = pd.read_csv("eval.csv")
    X_new = new_data[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].to_numpy()
    X_new = scaler.transform(X_new)  # 使用训练时的 scaler
    n_new_samples = X_new.shape[0] // timesteps
    X_new = X_new[:n_new_samples * timesteps].reshape(n_new_samples, timesteps, n_features)
    X_new = torch.FloatTensor(X_new)

    with torch.no_grad():
        X_new_pred = model(X_new)
        mse_new = torch.mean((X_new - X_new_pred) ** 2, dim=(1, 2)).numpy()
        anomalies = mse_new > threshold
        print("新数据的 MSE:", mse_new[:10])  # 显示前 10 个
        print("异常标记:", anomalies[:10])
        print("异常样本数:", np.sum(anomalies))
        
        # 验证准确率
        true_labels = new_data['ANOMALY_FLAG'].values[:n_new_samples * timesteps:timesteps]
        # print("真实标签:", true_labels[800:])
        accuracy = np.mean(anomalies == true_labels)
        print("异常检测准确率:", accuracy)

        # 添加 Matplotlib 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(mse_new, label='MSE', color='blue', alpha=0.5)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.6f})')
        plt.scatter(np.where(anomalies)[0], mse_new[anomalies], color='red', label='Anomalies', zorder=5)
        plt.title('MSE and Anomaly Detection')
        plt.xlabel('Sample Index')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        plt.savefig('./figures/mse_anomalies.png')  # 保存图形到文件
        plt.close()  # 关闭图形，避免内存占用
        print("MSE 和异常检测图已保存到 './figures/mse_anomalies.png'")

def cal_parms():
    model = TransformerAutoencoder(n_features=n_features, d_model=64, nhead=8, num_layers=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

if __name__ == "__main__":
    # 生成训练和评估数据
    # generate_data("train.csv")
    # generate_data("eval.csv")  # 这里应有不同数据，临时复用
    # train()
    predict()
    # cal_parms()