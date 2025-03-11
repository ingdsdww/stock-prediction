import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 定义数据集类
class StockDataset(Dataset):
    def __init__(self, labels_df, img_dir):
        self.labels_df = labels_df.head(400)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 转换为灰度图
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.labels_df.iloc[idx]['文件名']}"
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor(self.labels_df.iloc[idx]['收益率'], dtype=torch.float32)
        return image, label


class StockCNN(nn.Module):
    def __init__(self):
        super(StockCNN, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # 添加自适应平均池化层，将特征图大小固定
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 4 * 4, 128),  # 固定输入维度
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x * 10


#加载数据
def load_data():
    # 读取合并的CSV文件
    combined_df = pd.read_csv('all_stock_data.csv')
    
    # 按股票代码分组，转换为DataFrame列表
    data_list = []
    codes_list = []
    
    for code in combined_df['symbol'].unique():
        # 获取单个股票的数据
        stock_df = combined_df[combined_df['symbol'] == code].copy()
        # 按日期排序
        stock_df = stock_df.sort_values('date')
        # 删除symbol列（保持与原格式一致）
        stock_df = stock_df.drop('symbol', axis=1)
        
        data_list.append(stock_df)
        codes_list.append(code)
    
    return data_list, codes_list

# 加载数据
data, codes = load_data()
print(f"加载了 {len(data)} 只股票的数据")






def train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                patience=30, num_epochs=1000):
    # 初始化早停参数
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 记录训练历史
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs.squeeze(), labels).item()
        
        test_loss = test_loss/len(test_loader)
        test_losses.append(test_loss)
        
        # 打印当前epoch的训练和验证损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}')
        
        # 早停检查
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, test_losses

# 主程序
def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 读取标签数据
    labels_df = pd.read_excel('k_line_labels.xlsx')
    
    # 创建数据集
    full_dataset = StockDataset(labels_df, 'k_line_images')
    
    # 计算训练集和测试集的大小
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # 随机划分训练集和测试集
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 初始化模型
    model = StockCNN().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model, train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device,
        patience=30,  # 10个epoch没有改善就停止
        num_epochs=1000
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'stock_cnn_model.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


def evaluate_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    labels_df = pd.read_excel('k_line_labels.xlsx')
    # 取后100个数据
    labels_df = labels_df.tail(100)
    
    # 创建数据集
    dataset = StockDataset(labels_df, 'k_line_images')
    test_loader = DataLoader(dataset, batch_size=8)
    
    # 加载模型
    model = StockCNN().to(device)
    model.load_state_dict(torch.load('stock_cnn_model.pth'))
    model.eval()
    
    predictions = []
    actual_labels = []
    stock_codes = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().squeeze())
            actual_labels.extend(labels.numpy())
            
    # 创建预测结果DataFrame
    results_df = pd.DataFrame({
        '股票代码': labels_df['股票代码'].values,
        '实际收益率': actual_labels,
        '预测收益率': predictions,
        '预测误差': np.abs(np.array(predictions) - np.array(actual_labels))
    })
    
    print("\n预测结果统计：")
    print(results_df.describe())
    
    # 计算平均绝对误差
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_labels)))
    print(f"\n平均绝对误差 (MAE): {mae:.4f}")
    
    return results_df

# 运行评估
results = evaluate_model()
print("\n所有预测结果：")
print(results)