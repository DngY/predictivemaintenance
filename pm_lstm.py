import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle

# 生成数据
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

# LSTM 自编码器
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x: (batch_size, timesteps, input_size)
        # 编码
        _, (hidden, cell) = self.encoder(x)  # hidden: (num_layers, batch_size, hidden_size)
        
        # 重复隐藏状态以匹配时间步
        hidden = hidden[-1]  # 取最后一层: (batch_size, hidden_size)
        hidden = hidden.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch_size, timesteps, hidden_size)
        
        # 解码
        output, _ = self.decoder(hidden)  # (batch_size, timesteps, input_size)
        return output

# 参数
timesteps = 1
n_features = 4

# 添加噪声函数（不变）
def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(loc=0, scale=noise_factor * np.std(data, axis=0), size=data.shape)
    return data + noise

def train():
    print("......开始模型训练......")
    data_pd = pd.read_csv("train.csv")
    X = data_pd[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].iloc[:800].to_numpy()
    X_noisy = add_noise(X, noise_factor=0.01)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_noisy)
    with open("./models/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    n_samples = X_train.shape[0] // timesteps
    X_train = X_train[:n_samples * timesteps].reshape(n_samples, timesteps, n_features)
    X_train = torch.FloatTensor(X_train)
    print("X_train 形状:", X_train.shape)  # (800, 1, 4)

    model = LSTMAutoencoder(input_size=n_features, hidden_size=64, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    batch_size = 32
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

    torch.save(model.state_dict(), "./models/lstm_autoencoder.pth")
    print("模型已保存到 ./models/lstm_autoencoder.pth")

def predict():
    model = LSTMAutoencoder(input_size=n_features, hidden_size=64, num_layers=3)
    model.load_state_dict(torch.load("./models/lstm_autoencoder.pth"))
    model.eval()
    
    with open("./models/lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    data_pd = pd.read_csv("train.csv")
    X_train_full = data_pd[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].to_numpy()
    X_train_full = scaler.transform(X_train_full)
    n_samples = X_train_full.shape[0] // timesteps
    X_train = X_train_full[:n_samples * timesteps].reshape(n_samples, timesteps, n_features)
    X_train = torch.FloatTensor(X_train)
    
    with torch.no_grad():
        X_train_pred = model(X_train)
        mse_train = torch.mean((X_train - X_train_pred) ** 2, dim=(1, 2)).numpy()
        threshold = np.mean(mse_train) + 3 * np.std(mse_train)
        print("阈值:", threshold)

    new_data = pd.read_csv("eval.csv")
    X_new = new_data[['TEMPERATURE', 'VIBRATION', 'MOTOR_RPM', 'MOTOR_AMPS']].to_numpy()
    X_new = scaler.transform(X_new)
    n_new_samples = X_new.shape[0] // timesteps
    X_new = X_new[:n_new_samples * timesteps].reshape(n_new_samples, timesteps, n_features)
    X_new = torch.FloatTensor(X_new)

    with torch.no_grad():
        X_new_pred = model(X_new)
        mse_new = torch.mean((X_new - X_new_pred) ** 2, dim=(1, 2)).numpy()
        anomalies = mse_new > threshold
        print("新数据的 MSE（前 10 个）:", mse_new[:10])
        print("异常标记（前 10 个）:", anomalies[:10])
        print("新数据的 MSE（后 10 个）:", mse_new[-10:])
        print("异常标记（后 10 个）:", anomalies[-10:])
        print("异常样本数:", np.sum(anomalies))
        
        true_labels = new_data['ANOMALY_FLAG'].values
        assert len(true_labels) == len(anomalies), "标签和预测长度不匹配"
        accuracy = np.mean(anomalies == true_labels)
        print("异常检测准确率:", accuracy)


def cal_parms():
    model = LSTMAutoencoder(input_size=4, hidden_size=64, num_layers=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

if __name__ == "__main__":
    # generate_data("train.csv")
    # generate_data("eval.csv")  # 临时复用，实际应生成不同数据
    # train()
    predict()
    # cal_parms()