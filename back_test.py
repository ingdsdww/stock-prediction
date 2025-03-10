
from cnn_train import StockCNN
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_save_candlestick(df, stock_code, save_path='temp_images'):
    # 创建保存图片的文件夹
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 处理数据，提取需要的列
    df = df[['date', 'open', 'close', 'high', 'low']]
    df.set_index('date', inplace=True)
    
    # 创建图形对象
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算K线图的宽度
    width = 0.6
    
    # 绘制K线图
    for index, row in df.iterrows():
        if row['close'] >= row['open']:  # 上涨
            color = 'red'
            bottom = row['open']
            height = row['close'] - row['open']
        else:  # 下跌
            color = 'green'
            bottom = row['close']
            height = row['open'] - row['close']
        
        ax.bar(index, height, width, bottom=bottom, color=color)
        ax.plot([index, index], [row['low'], row['high']], color=color, linewidth=1)
    
    # 移除所有图例和标签
    ax.set_xticks([])  # 移除x轴刻度
    ax.set_yticks([])  # 移除y轴刻度
    ax.set_title('')   # 移除标题
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 保存图片，设置紧凑布局
    plt.savefig(f'{save_path}/{stock_code}.png', 
                bbox_inches='tight', 
                pad_inches=0,  # 移除所有边距
                dpi=100)
    plt.close()


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

data,codes = load_data()
#加载模型并预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockCNN().to(device)
model.load_state_dict(torch.load('stock_cnn_model.pth'))

def backtest_strategy(data_list, model, device, start_date, end_date, top_k=5):
    class SingleStockDataset(Dataset):
        def __init__(self, df):
            self.df = df
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            temp_dir = 'temp_images'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            plot_save_candlestick(self.df, 'temp', temp_dir)
            image = Image.open(f'{temp_dir}/temp.png')
            image = self.transform(image)
            return image, torch.tensor(0.0)

    def predict_return(hist_data):
        """预测单只股票的次日收益率"""
        try:
            dataset = SingleStockDataset(hist_data)
            loader = DataLoader(dataset, batch_size=1)
            
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    predicted_return = model(inputs)
                    return predicted_return.item()
        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            return None

    # 预处理日期
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 预处理所有股票数据
    processed_data = []
    for df in data_list:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if len(df) >= 20:  # 确保有足够的历史数据
            processed_data.append(df)
    
    # 获取所有交易日
    trading_days = sorted(set(pd.concat([df['date'] for df in processed_data])))
    
    # 初始化变量
    initial_capital = 1000000
    capital = initial_capital
    daily_returns = []
    trade_records = []  # 记录所有交易
    positions = {}      # 记录当前持仓 {symbol: {'price': price, 'type': 'long/short', 'date': date}}
    
    # 预处理日期
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 获取所有交易日
    trading_days = set()
    for df in data_list:
        df['date'] = pd.to_datetime(df['date'])
        trading_days.update(df['date'].tolist())
    trading_days = sorted([day for day in trading_days if start_date <= day <= end_date])
    
    print(f"交易日数量: {len(trading_days)}")
    
    # 对每个交易日进行回测
    for i, current_date in enumerate(trading_days[:-1]):
        next_date = trading_days[i + 1]
        predictions = []
        
        # 对每只股票进行预测
        for j, df in enumerate(data_list):
            try:
                hist_data = df[df['date'] <= current_date].copy()
                if len(hist_data) < 20:
                    continue
                
                pred_return = predict_return(hist_data)
                if pred_return is None:
                    continue
                    
                current_price = df[df['date'] == current_date]['close'].iloc[0]
                next_price = df[df['date'] == next_date]['close'].iloc[0]
                actual_return = (next_price / current_price - 1)
                
                predictions.append({
                    'symbol': codes[j],
                    'pred_return': float(pred_return),
                    'actual_return': float(actual_return * 100),
                    'close_price': float(current_price)
                })
            except Exception as e:
                continue
        
        if len(predictions) > 0:
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.sort_values('pred_return', ascending=False)
            
            # 选择股票
            new_long_positions = set(pred_df.head(top_k)['symbol'])
            new_short_positions = set(pred_df.tail(top_k)['symbol'])
            
            # 处理平仓
            for symbol in list(positions.keys()):
                if (positions[symbol]['type'] == 'long' and symbol not in new_long_positions) or \
                   (positions[symbol]['type'] == 'short' and symbol not in new_short_positions):
                    # 获取平仓价格
                    df = data_list[codes.index(symbol)]
                    close_price = df[df['date'] == current_date]['close'].iloc[0]
                    entry_price = positions[symbol]['price']
                    entry_date = positions[symbol]['date']
                    pos_type = positions[symbol]['type']
                    
                    # 计算收益
                    if pos_type == 'long':
                        profit = (close_price - entry_price) / entry_price * 100
                    else:  # short
                        profit = (entry_price - close_price) / entry_price * 100
                    
                    # 记录交易
                    trade_records.append({
                        'symbol': symbol,
                        'type': pos_type,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': close_price,
                        'profit_pct': profit
                    })
                    
                    # 移除持仓
                    del positions[symbol]
            
            # 处理开仓
            for symbol in new_long_positions:
                if symbol not in positions:
                    df = data_list[codes.index(symbol)]
                    entry_price = df[df['date'] == current_date]['close'].iloc[0]
                    positions[symbol] = {
                        'price': entry_price,
                        'type': 'long',
                        'date': current_date
                    }
                    
                    trade_records.append({
                        'symbol': symbol,
                        'type': 'long',
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'exit_date': None,
                        'exit_price': None,
                        'profit_pct': None
                    })
            
            for symbol in new_short_positions:
                if symbol not in positions:
                    df = data_list[codes.index(symbol)]
                    entry_price = df[df['date'] == current_date]['close'].iloc[0]
                    positions[symbol] = {
                        'price': entry_price,
                        'type': 'short',
                        'date': current_date
                    }
                    
                    trade_records.append({
                        'symbol': symbol,
                        'type': 'short',
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'exit_date': None,
                        'exit_price': None,
                        'profit_pct': None
                    })
            # 计算当日收益率
            long_returns = [pred_df[pred_df['symbol'] == sym]['actual_return'].iloc[0] for sym in new_long_positions]
            short_returns = [-pred_df[pred_df['symbol'] == sym]['actual_return'].iloc[0] for sym in new_short_positions]
            daily_return = (np.mean(long_returns) + np.mean(short_returns)) / 2 if long_returns and short_returns else 0
            # 记录每日结果
            daily_returns.append({
                'date': next_date,
                'daily_return': float(daily_return),
                'capital': float(capital),
                'long_positions': ','.join(new_long_positions),
                'short_positions': ','.join(new_short_positions),
                'position_count': len(positions)
            })
    
    # 转换为DataFrame
    returns_df = pd.DataFrame(daily_returns)
    trades_df = pd.DataFrame(trade_records)
    
    # 计算交易统计
    if len(trades_df) > 0:
        completed_trades = trades_df[trades_df['exit_date'].notna()]
        print("\n====== 交易统计 ======")
        print(f"总交易次数: {len(trades_df)}")
        print(f"完成交易次数: {len(completed_trades)}")
        print(f"平均持仓时间: {(completed_trades['exit_date'] - completed_trades['entry_date']).mean().days:.1f}天")
        print(f"胜率: {(completed_trades['profit_pct'] > 0).mean()*100:.2f}%")
        print(f"平均盈利: {completed_trades[completed_trades['profit_pct'] > 0]['profit_pct'].mean():.2f}%")
        print(f"平均亏损: {completed_trades[completed_trades['profit_pct'] < 0]['profit_pct'].mean():.2f}%")
    
    # 保存交易记录
    trades_df.to_csv('trade_records.csv', index=False)
    returns_df.to_csv('daily_returns.csv', index=False)
    
    return returns_df, trades_df

# 运行回测
start_date = '2023-01-01'
end_date = '2023-12-31'
results = backtest_strategy(data, model, device, start_date, end_date, top_k=3)