import pandas as pd
import akshare as ak
#读取股票列表获得股票代码
list = pd.read_excel('股票列表.xlsx')
#从list获取所有代码
codes = list[list['代码'].str.startswith('bj') == False]['代码'].tolist()
# 筛选数据
filtered_codes = [code for code in codes if not (code[-6:].startswith('300') or code[-6:].startswith('301'))]
#获得全部股票日线数据
data = []
j = 0
for i in codes:
    data.append(ak.stock_zh_index_daily_em(symbol=i, start_date="20240609", end_date="20250302"))
    j = j + 1
    if j == 1100:
        break


#画图函数
import matplotlib.pyplot as plt

def plot_save_candlestick(df, stock_code, start_date="20240609",end_date="20250302", save_path='k_line_images'):
    df['date'] = pd.to_datetime(df['date'])
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]
    # 计算最后一天的收益率
    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    return_rate = (last_day['close'] - prev_day['close']) / prev_day['close'] * 100
    
    # 移除最后一天的数据再绘图
    df = df[:-1]
    
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
    
    return return_rate

# 创建列表存储图片信息
image_data = []

# 批量绘制并保存K线图
for i, df in enumerate(data):
    try:
        if len(df) >= 2:  # 确保至少有两天的数据
            return_rate = plot_save_candlestick(df, codes[i])
            image_data.append({
                '图片编号': i,
                '股票代码': codes[i],
                '文件名': f'{codes[i]}.png',
                '收益率': return_rate
            })
            
            if i % 100 == 0:  # 每处理100张图片打印一次进度
                print(f"已处理 {i} 张图片")

    except Exception as e:
        print(f"处理股票 {codes[i]} 时出错: {str(e)}")
    
# 将图片信息保存为Excel
image_df = pd.DataFrame(image_data)
image_df.to_excel('k_line_labels.xlsx', index=False)

# 将所有股票数据合并为一个DataFrame并保存
all_data = []

for i, df in enumerate(data):
    # 添加股票代码列
    df['symbol'] = codes[i]
    all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 保存为单个CSV文件
combined_df.to_csv('all_stock_data.csv', index=False)

print(f"合并后的数据形状: {combined_df.shape}")
print("\n数据示例:")
print(combined_df.head())