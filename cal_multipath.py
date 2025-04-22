import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def calculate_pseudo_noise(df):
    """计算伪距噪声，根据可用数据列灵活计算"""
    # 定义波长常数（单位：米）
    wavelength_L1 = 299792458 / 1575.42e6  # GPS L1信号波长
    wavelength_L5 = 299792458 / 1176.45e6  # GPS L5信号波长

    # 检查并计算L1波段噪声
    if 'C1C' in df.columns and 'L1C' in df.columns:
        df['C1_minus_L1B'] = df['C1C'] - df['L1C'] * wavelength_L1
        df['C1C_noise'] = df['C1_minus_L1B'].diff()
        df.loc[df.index[0], 'C1C_noise'] = 0

    # 检查并计算L5波段噪声
    if 'C5Q' in df.columns and 'L5Q' in df.columns:
        df['C5_minus_L5B'] = df['C5Q'] - df['L5Q'] * wavelength_L5
        df['C5Q_noise'] = df['C5_minus_L5B'].diff()
        df.loc[df.index[0], 'C5Q_noise'] = 0

    return df



def remove_outliers(series, n_std=3):
    """
    使用标准差方法检测和剔除离群值
    series: 数据序列
    n_std: 标准差的倍数阈值（默认为3）
    返回: 处理后的序列，以及异常值的索引
    """
    mean = series.mean()
    std = series.std()
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    # 找出异常值的索引
    outlier_idx = series[(series < lower_bound) | (series > upper_bound)].index

    # 将异常值设置为NaN
    series_clean = series.copy()
    series_clean[outlier_idx] = np.nan

    return series_clean, outlier_idx


def calculate_mw_combination(df):
    """计算Melbourne-Wübbena组合值"""
    # 定义频率常数
    f1 = 1575.42e6  # L1频率（Hz）
    f2 = 1176.45e6  # L5频率（Hz）

    # 计算波长
    c = 299792458  # 光速（m/s）
    lambda1 = c / f1  # L1波长（m）
    lambda2 = c / f2  # L5波长（m）

    # 计算MW组合的系数
    freq_factor = (f1 - f2) / (f1 + f2)

    # 计算MW组合值
    df['MW'] = (freq_factor * (df['P1'] / lambda1 + df['P2'] / lambda2) -
                (df['L1'] - df['L2']))

    return df['MW']


def detect_cycle_slips(df):
    """使用MW组合检测周跳"""
    # 定义频率常数
    f1 = 1575.42e6  # L1频率（Hz）
    f2 = 1176.45e6  # L5频率（Hz）

    # 计算波长
    c = 299792458  # 光速（m/s）
    lambda1 = c / f1  # L1波长（m）
    lambda2 = c / f2  # L5波长（m）

    freq_factor = lambda1*lambda2/(lambda2-lambda1)

    mw = freq_factor*(df['L1C']-df['L5Q'])-((lambda2*df['C5Q']+lambda1*df['C1C'])/(lambda1+lambda2))
    # # 计算MW组合的系数
    # freq_factor = (f1 - f2) / (f1 + f2)
    #
    # # 计算MW组合值
    # mw = (freq_factor * (df['C1C'] / lambda1 + df['C5Q'] / lambda2) -
    #       (df['L1C'] - df['L5Q']))

    # 计算MW组合值的差分
    mw_diff = mw.diff()

    # 设置MW组合周跳检测阈值（可根据实际情况调整）
    threshold = 5.0  # 单位：周

    # 检测周跳点
    slip_points = []
    for i in range(1, len(df)):
        if abs(mw_diff.iloc[i]) > threshold:
            slip_points.append(i)
            print(f"检测到周跳点：历元 {i}, MW组合跳变值：{mw_diff.iloc[i]:.3f}")

    # 添加起始点和终点
    slip_points = [0] + slip_points + [len(df)]
    return sorted(list(set(slip_points)))  # 去除重复点并排序


def remove_outliers_mad(series, k=3.0):
    """
    使用中位数绝对偏差(MAD)方法检测和剔除离群值
    这是一种比基于均值和标准差更稳健的方法

    参数:
    series: pandas.Series, 输入数据序列
    k: float, MAD的倍数阈值(默认为3.0)

    返回:
    series_clean: 处理后的序列，异常值被替换为NaN
    outlier_idx: 异常值的索引
    """
    # 计算中位数
    median = series.median()

    # 计算MAD
    mad = (series - median).abs().median()

    # 标准化MAD (乘以1.4826使其与正态分布的标准差可比)
    mad_normalized = 1.4826 * mad

    # 计算离中位数的距离阈值
    threshold = k * mad_normalized

    # 找出异常值的索引
    outlier_idx = series[((series - median).abs() > threshold)].index

    # 将异常值设置为NaN
    series_clean = series.copy()
    series_clean[outlier_idx] = np.nan

    return series_clean, outlier_idx


def calculate_multipath(df):
    """计算双频伪距多路径误差，使用MW组合检测周跳，并进行基于MAD的异常值检测"""
    # 定义频率常数
    f1 = 1575.42e6  # L1频率（Hz）
    f2 = 1176.45e6  # L5频率（Hz）

    # 计算波长
    c = 299792458  # 光速（m/s）
    lambda1 = c / f1  # L1波长（m）
    lambda2 = c / f2  # L5波长（m）

    # 计算频率项系数
    f1_square = f1 * f1
    f2_square = f2 * f2

    # 计算公式中的系数
    # coef1_P1 = -(f1_square + f2_square) / (f1_square - f2_square)
    # coef2_P1 = 2 * f2_square / (f1_square - f2_square)
    #
    # coef1_P2 = -2 * f2_square / (f1_square - f2_square)
    # coef2_P2 = (f1_square + f2_square) / (f1_square - f2_square)

    # test
    coef1_P1 = f1_square / f2_square


    # 检查必要的观测值是否存在
    required_cols = ['C1C', 'L1C', 'C5Q', 'L5Q']
    if not all(col in df.columns for col in required_cols):
        print("警告：缺少计算多路径所需的观测值")
        return df

    # 计算原始MP1和MP2
    # df['MP1'] = (df['C1C'] + (coef1_P1 * lambda1 * df['L1C'] + coef2_P1 * lambda2 * df['L5Q']))
    # df['MP2'] = (df['C5Q'] - (coef1_P2 * lambda1 * df['L1C'] + coef2_P2 * lambda2 * df['L5Q']))

    df['MP1'] = df['C1C'] + (1-coef1_P1)/(1+coef1_P1)* lambda1 *df['L1C'] - 2/(1-coef1_P1)* lambda2 *df['L5Q']
    df['MP2'] = df['C1C'] + (2*coef1_P1)/(1-coef1_P1)* lambda1 *df['L1C'] - (1+coef1_P1)/(1-coef1_P1)* lambda2 *df['L5Q']


    # 使用MW组合检测周跳点
    slip_points = detect_cycle_slips(df)

    # 在每个区间内分别处理
    for i in range(len(slip_points) - 1):
        start_idx = slip_points[i]
        end_idx = slip_points[i + 1]

        # 检查区间长度
        if end_idx - start_idx < 10:  # 如果区间太短，可能是虚警
            continue

        # 获取当前区间的MP1和MP2数据
        mp1_segment = df['MP1'].iloc[start_idx:end_idx]
        mp2_segment = df['MP2'].iloc[start_idx:end_idx]

        # 使用MAD方法进行异常值检测
        mp1_clean, mp1_outliers = remove_outliers_mad(mp1_segment, k=3.0)
        mp2_clean, mp2_outliers = remove_outliers_mad(mp2_segment, k=3.0)

        # 打印异常值信息
        if len(mp1_outliers) > 0:
            print(f"区间 {i + 1} MP1异常值数量: {len(mp1_outliers)}")
            print(f"MP1异常值范围: [{mp1_segment[mp1_outliers].min():.2f}, {mp1_segment[mp1_outliers].max():.2f}]")
        if len(mp2_outliers) > 0:
            print(f"区间 {i + 1} MP2异常值数量: {len(mp2_outliers)}")
            print(f"MP2异常值范围: [{mp2_segment[mp2_outliers].min():.2f}, {mp2_segment[mp2_outliers].max():.2f}]")

        # 计算清理后数据的中位数（而不是平均值）
        mp1_median = mp1_clean.median()
        mp2_median = mp2_clean.median()

        # 更新原始数据（使用中位数进行归中）
        df.loc[df.index[start_idx:end_idx], 'MP1'] = mp1_clean - mp1_median
        df.loc[df.index[start_idx:end_idx], 'MP2'] = mp2_clean - mp2_median

        # 打印区间信息
        interval_length = end_idx - start_idx
        print(f"区间 {i + 1}: 起点={start_idx}, 终点={end_idx}, 长度={interval_length}")
        print(f"      MP1中位数={mp1_median:.3f}m, MP2中位数={mp2_median:.3f}m")

    return df


def process_single_file(file_path):
    """处理单个文件并返回处理后的数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, delimiter=',')
        df.columns = df.columns.str.strip()

        # 找到时间列和高度角列
        time_column = next((col for col in df.columns if 'time' in col.lower()), None)
        el_column = next((col for col in df.columns if 'el' in col.lower()), None)

        if not time_column or not el_column:
            print(f"警告: {file_path} 缺少必要的列")
            return None

        # 转换时间格式和高度角数据
        df['datetime'] = pd.to_datetime(df[time_column])
        df['el'] = pd.to_numeric(df[el_column], errors='coerce')

        # 检查并映射可用的列
        required_pairs = [('C1C', 'L1C'), ('C5Q', 'L5Q')]
        column_mapping = {}

        for pair in required_pairs:
            pair_cols = [next((col for col in df.columns if req in col), None) for req in pair]
            if all(col is not None for col in pair_cols):
                for orig_col, req_col in zip(pair_cols, pair):
                    column_mapping[orig_col] = req_col

        if not column_mapping:
            print(f"警告: {file_path} 没有找到可用的观测值列对")
            return None

        # 重命名列并转换数据类型
        df = df.rename(columns=column_mapping)
        for col in column_mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算伪距噪声
        df = calculate_pseudo_noise(df)

        # 添加文件名标识（提取卫星编号）
        df['satellite'] = Path(file_path).stem

        return df

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")
        return None


def plot_pseudo_noise_time_multi(data_frames, title=None):
    """绘制多卫星伪距噪声随时间变化的图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个卫星分配一个标记
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}

    # 固定波段的颜色
    c1_color = 'blue'
    c5_color = 'red'

    # 给C5波段添加小的偏移量，使其更容易看到
    # offset = 0.5

    # 先绘制所有C1波段（设置较低的透明度）
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]
        if 'C1C_noise' in df.columns:
            ax.plot(df['C1C_noise'][1:].values, color=c1_color,
                    linestyle='-', linewidth=1, marker=marker, markersize=1.2,
                    label=f'{sat_name}-C1波段', alpha=0.4)
            print(f"{sat_name}-C1波段: 标准差={df['C1C_noise'][1:].std():.3f}m")

    # 再绘制所有C5波段（增加偏移量并设置较高的透明度）
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]
        if 'C5Q_noise' in df.columns:
            # 添加偏移量
            c5_data = df['C5Q_noise'][1:].values
            ax.plot(c5_data, color=c5_color,
                    linestyle='-', linewidth=1, marker=marker, markersize=1.2,
                    label=f'{sat_name}-C5波段', alpha=0.7)
            print(f"{sat_name}-C5波段: 标准差={df['C5Q_noise'][1:].std():.3f}m")

    # 设置图表属性
    ax.set_xlabel('UTC(hh:mm)', fontsize=12)
    ax.set_ylabel('伪距噪声(m)', fontsize=12)
    ax.set_ylim(-40, 40)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 设置时间轴
    sample_df = next(iter(data_frames.values()))
    times = sample_df['datetime'][1:]
    num_points = len(times)
    tick_interval = num_points // 10 if num_points > 1000 else num_points // 5
    xticks = np.arange(0, num_points, tick_interval)
    ax.set_xticks(xticks)

    if hasattr(times, 'dt'):
        time_labels = [times.iloc[i].strftime('%H:%M') if i < len(times) else '' for i in xticks]
        ax.set_xticklabels(time_labels, rotation=30, ha='right')

    ax.set_facecolor('white')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pseudo_noise_elevation_multi(data_frames, title=None):
    """绘制多卫星伪距噪声随高度角变化的图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个卫星分配一个标记
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}

    # 固定波段的颜色
    c1_color = 'blue'
    c5_color = 'red'

    # 绘制所有卫星的所有波段数据
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]

        # 绘制C1波段
        if 'C1C_noise' in df.columns:
            ax.scatter(df['el'][1:], df['C1C_noise'][1:],
                       c=c1_color, marker=marker, s=10, alpha=0.6,
                       label=f'{sat_name}-C1波段')

        # 绘制C5波段
        if 'C5Q_noise' in df.columns:
            ax.scatter(df['el'][1:], df['C5Q_noise'][1:],
                       c=c5_color, marker=marker, s=10, alpha=0.6,
                       label=f'{sat_name}-C5波段')

    # 设置图表属性
    ax.set_xlabel('卫星高度角(°)', fontsize=12)
    ax.set_ylabel('伪距噪声(m)', fontsize=12)
    ax.set_ylim(-40, 40)
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_facecolor('white')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multipath(data_frames, mp_type='MP1', title=None):
    """绘制单个多路径误差图像"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个卫星分配一个标记
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}

    # 绘制多路径误差
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]
        if mp_type in df.columns:
            ax.plot(df[mp_type].values,
                    linestyle='-', linewidth=0.8, marker=marker, markersize=1.2,
                    label=f'{sat_name}-{mp_type}', alpha=0.7)
            print(f"{sat_name}-{mp_type}: 标准差={df[mp_type].std():.3f}m")

    # 设置图表属性
    ax.set_xlabel('历元', fontsize=12)
    ax.set_ylabel('多路径误差(m)', fontsize=12)
    ax.set_ylim(-10, 10)  # 根据实际情况调整范围
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_facecolor('white')

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_multipath_el(data_frames, mp_type='MP1', title=None):
    """绘制单个多路径误差随高度角变化的散点图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个卫星分配一个标记
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}

    # 绘制多路径误差随高度角的变化
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]
        if mp_type in df.columns:
            ax.scatter(df['el'], df[mp_type],
                       marker=marker, s=10, alpha=0.6,
                       label=f'{sat_name}-{mp_type}')

    # 设置图表属性
    ax.set_xlabel('卫星高度角(°)', fontsize=12)
    ax.set_ylabel('多路径误差(m)', fontsize=12)
    ax.set_ylim(-10, 10)  # 根据实际情况调整范围
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_facecolor('white')

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def main():
    """主函数 - 处理数据并生成图表"""
    # 指定数据文件夹路径
    folder_path = r'C:\Users\zhang\Desktop\paper\data_analysis\P40\GPS'

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print("未找到CSV文件")
        return

    # 处理所有文件
    processed_data = {}
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"正在处理文件: {file_name}")

        df = process_single_file(file_path)
        if df is not None:
            # 计算多路径误差
            df = calculate_multipath(df)
            # 使用文件名（卫星编号）作为键
            satellite_id = Path(file_name).stem
            processed_data[satellite_id] = df

    if not processed_data:
        print("没有成功处理任何文件")
        return

    # 分别绘制MP1和MP2的图形
    plot_multipath(processed_data, 'MP1', "L1频段多路径误差时间序列")
    plot_multipath(processed_data, 'MP2', "L5频段多路径误差时间序列")
    # plot_multipath_el(processed_data, 'MP1', "L1频段多路径误差-高度角关系")
    # plot_multipath_el(processed_data, 'MP2', "L5频段多路径误差-高度角关系")
    plot_pseudo_noise_elevation_multi(processed_data)
    plot_pseudo_noise_time_multi(processed_data)


if __name__ == "__main__":
    main()