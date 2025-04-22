import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
import matplotlib as mpl


# 设置中文字体
def set_chinese_font():
    # 检查宋体是否可用
    font_names = [f.name for f in fm.fontManager.ttflist]

    # 中文字体设置
    if 'SimSun' in font_names:
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
    elif 'SimHei' in font_names:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体作为备选
    else:
        print("警告: 未找到宋体或黑体，使用系统默认字体")

    # 设置英文字体为Times New Roman
    if 'Times New Roman' in font_names:
        plt.rcParams['font.serif'] = ['Times New Roman']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


def read_satellite_data(file_path):
    """
    读取卫星数据文件并解析为结构化数据，支持多历元

    Args:
        file_path: 文件路径

    Returns:
        dict: 包含每个历元数据的字典，格式为 {timestamp: DataFrame}
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # 按历元分割数据
    epoch_sections = content.split('>')

    # 过滤掉空部分和只包含注释的部分
    epoch_sections = [section for section in epoch_sections[1:] if section.strip()]  # 跳过第一部分（可能包含注释）

    epochs_data = {}

    for section in epoch_sections:
        lines = section.strip().split('\n')

        # 第一行包含时间戳
        timestamp = lines[0].strip()

        data = []

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # 使用正则表达式分割字段，处理多个空格
            parts = re.split(r'\s+', line.strip())

            if len(parts) >= 8:  # 确保有足够的列
                satellite_id = parts[0]
                freq = parts[1]

                # 安全地提取数值 - 正确映射列名
                try:
                    prior = float(parts[2])  # 第3列是Prior数据
                except ValueError:
                    prior = np.nan

                try:
                    post = float(parts[3])  # 第4列是Post数据
                except ValueError:
                    post = np.nan

                try:
                    prior_r = float(parts[4])  # 第5列是PriorR数据
                except ValueError:
                    prior_r = np.nan

                try:
                    post_r = float(parts[5])  # 第6列是PostR数据
                except ValueError:
                    post_r = np.nan

                try:
                    elevation = float(parts[6]) if parts[6] != "0.00" else np.nan
                except ValueError:
                    elevation = np.nan

                try:
                    snr = float(parts[7]) if len(parts) > 7 and parts[7] != "0.00" else np.nan
                except ValueError:
                    snr = np.nan

                # 卫星系统（通过卫星ID前缀判断）
                if satellite_id.startswith('G'):
                    system = 'GPS'
                elif satellite_id.startswith('C'):
                    system = 'BeiDou'
                elif satellite_id.startswith('R'):
                    system = 'GLONASS'
                else:
                    system = 'Other'

                data.append({
                    'Timestamp': timestamp,
                    'SatelliteID': satellite_id,
                    'Frequency': freq,
                    'System': system,
                    'Prior': prior,
                    'Post': post,
                    'PriorR': prior_r,
                    'PostR': post_r,
                    'Elevation': elevation,
                    'SNR': snr
                })

        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 将时间戳转换为datetime对象
        try:
            timestamp_dt = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S.%f')
        except ValueError:
            try:
                timestamp_dt = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
            except ValueError:
                timestamp_dt = None

        if timestamp_dt:
            epochs_data[timestamp_dt] = df

    return epochs_data


def plot_system_data(valid_data, system_name, plot_type, prior_threshold=150):
    """
    绘制特定卫星系统的数据图表，仿照参考样式

    Args:
        valid_data: 有效数据DataFrame
        system_name: 卫星系统名称
        plot_type: 图表类型 ('elevation' 或 'snr')
        prior_threshold: Post数据的阈值，用于设置y轴范围
    """
    system_data = valid_data[valid_data['System'] == system_name]

    if system_data.empty:
        print(f"警告: {system_name}系统没有有效数据")
        return

    # 创建图表
    plt.figure(figsize=(10, 8))

    # 设置网格样式（虚线网格，类似参考图片）
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)

    # 绘制散点图
    if plot_type == 'elevation':
        x_data = system_data['Elevation']
        x_label = '高度角 (°)'
        x_lim = (5, 85)
    else:  # SNR图
        x_data = system_data['SNR']
        x_label = '载噪比 (dB-Hz)'
        x_lim = (10, 50)

    # 取Post的绝对值
    y_data = system_data['Post'].abs()

    # 去除x和y中的NaN值
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_clean = x_data[mask].values
    y_clean = y_data[mask].values

    # 检查是否有足够的有效数据点
    if len(x_clean) > 5:
        # 绘制散点图
        plt.scatter(x_clean, y_clean, color='blue', s=5, alpha=0.7, label='伪距残差')

        try:
            # 使用更稳健的曲线拟合方法
            # 首先按x值排序
            sort_idx = np.argsort(x_clean)
            x_sorted = x_clean[sort_idx]
            y_sorted = y_clean[sort_idx]

            # 检查数据是否有大的方差
            if np.var(y_sorted) > 0.001:  # 确保数据有足够的变异性
                # 尝试二次多项式拟合
                z = np.polyfit(x_sorted, y_sorted, 2)
                p = np.poly1d(z)

                # 创建更密集的点以绘制平滑曲线
                x_dense = np.linspace(np.min(x_sorted), np.max(x_sorted), 100)
                y_dense = p(x_dense)

                # 绘制拟合曲线
                plt.plot(x_dense, y_dense, 'r-', linewidth=2, label='拟合曲线')

                # 显示拟合公式
                # equation = f"y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}"
                plt.annotate(xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            else:
                print(f"警告: {system_name}系统的{plot_type}数据变异性太小，不适合拟合曲线")
        except Exception as e:
            print(f"警告: 无法为{system_name}系统的{plot_type}数据创建拟合曲线: {str(e)}")
    else:
        # 仅绘制散点图
        plt.scatter(x_data, y_data, color='blue', s=5, alpha=0.7, label='伪距残差')
        print(f"警告: {system_name}系统的{plot_type}数据点数不足（{len(x_clean)}个有效点），无法拟合曲线")

    plt.xlabel(x_label, fontsize=10)
    plt.ylabel('伪距残差 (m)', fontsize=10)
    plt.title(f'{system_name}', fontsize=10)

    # 设置坐标轴范围
    plt.xlim(x_lim)
    plt.ylim(0, prior_threshold)

    # 设置图例，放在右上角
    plt.legend(loc='upper right', fontsize=10, frameon=True)

    # 设置坐标轴样式
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 使坐标轴线更粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 紧凑布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def plot_all_systems(epochs_data, prior_threshold=150):
    """
    按照卫星系统分别绘制数据图表

    Args:
        epochs_data: 包含每个历元数据的字典
        prior_threshold: Post数据的阈值，超过此值的数据将被排除（默认为150）
    """
    # 设置中文字体
    set_chinese_font()

    # 按时间排序历元
    sorted_epochs = sorted(epochs_data.keys())

    # 确保有历元数据
    if not sorted_epochs:
        print("警告: 没有找到任何历元数据")
        return

    # 跳过第一个历元（如果有多个历元）
    if len(sorted_epochs) > 1:
        sorted_epochs = sorted_epochs[1:]
    else:
        print("警告: 只有一个历元，无法跳过第一个历元")
        return

    print(f"将绘制以下历元的数据: {[epoch.strftime('%Y-%m-%d %H:%M:%S') for epoch in sorted_epochs]}")

    # 合并除第一个历元外的所有数据
    combined_data = pd.DataFrame()
    for timestamp in sorted_epochs:
        df = epochs_data[timestamp]
        # 添加一个时间标识列，用于区分不同历元
        df['EpochTime'] = timestamp.strftime('%H:%M:%S')
        combined_data = pd.concat([combined_data, df])

    # 显示合并数据的基本信息
    print(f"合并数据总行数: {len(combined_data)}")
    print(f"合并数据中的历元: {combined_data['EpochTime'].unique()}")
    print(f"合并数据中的卫星系统: {combined_data['System'].unique()}")

    # 过滤有效数据（去除Elevation或SNR为NaN的行）
    valid_data = combined_data.dropna(subset=['Elevation', 'SNR']).copy()

    # 确保Post列有数据
    valid_data = valid_data.dropna(subset=['Post']).copy()

    # 创建Post绝对值列(但不修改原始数据)
    valid_data['Post_Abs'] = valid_data['Post'].abs()

    # 筛选出Post绝对值小于阈值的数据
    valid_data = valid_data[valid_data['Post_Abs'] <= prior_threshold].copy()

    print(f"应用Post阈值({prior_threshold})后的有效数据行数: {len(valid_data)}")

    if valid_data.empty:
        print("警告: 没有有效数据可供绘制")
        return

    # 获取所有存在的卫星系统
    systems = valid_data['System'].unique()

    # 为每个系统分别绘制Elevation和SNR图
    for system in systems:
        # 绘制Post vs Elevation图
        plot_system_data(valid_data, system, 'elevation', prior_threshold)

        # 绘制Post vs SNR图
        plot_system_data(valid_data, system, 'snr', prior_threshold)


# 修改主函数，确保正确传递阈值
def main(file_path, prior_threshold=150):
    # 读取数据
    epochs_data = read_satellite_data(file_path)

    print(f"共读取 {len(epochs_data)} 个历元数据")
    print(f"设定的Post阈值: {prior_threshold} (Post绝对值大于此值的数据将被排除)")

    # 显示每个历元的数据概览
    for i, (timestamp, df) in enumerate(sorted(epochs_data.items())):
        print(f"\n历元 {i + 1}: {timestamp}")
        print(f"数据行数: {len(df)}")
        print(f"卫星系统分布: {df['System'].value_counts().to_dict()}")

        # 检查是否有有效的Elevation和SNR数据
        valid_count = df.dropna(subset=['Elevation', 'SNR', 'Post']).shape[0]
        print(f"有效数据行数（含有效的Elevation、SNR和Post值）: {valid_count}")

    # 绘制所有系统的图表
    plot_all_systems(epochs_data, prior_threshold)


# 使用示例
if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='卫星数据分析与可视化')
    parser.add_argument('--file', type=str, default="C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\p40_res_dynamic.txt", help='数据文件路径')
    parser.add_argument('--threshold', type=float, default=150, help='Prior数据绝对值阈值，超过此值的数据将被排除')

    # 解析命令行参数
    args = parser.parse_args()

    # 运行主函数
    main(args.file, args.threshold)

# # 使用示例
# if __name__ == "__main__":
#     file_path = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\p40_res_dynamic.txt"  # 替换为实际文件路径
#     main(file_path)