import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from pathlib import Path


def calculate_pseudo_noise(df):
    """计算伪距噪声，根据可用数据列灵活计算"""
    wavelength_L1 = 299792458 / 1561.098e6  # GPS L1信号波长
    wavelength_L5 = 299792458 / 1176.45e6  # GPS L5信号波长

    if 'C1C' in df.columns and 'L1C' in df.columns:
        df['C1_minus_L1B'] = df['C1C'] - df['L1C'] * wavelength_L1
        df['C1C_noise'] = df['C1_minus_L1B'].diff()
        df.loc[df.index[0], 'C1C_noise'] = 0

    if 'C5Q' in df.columns and 'L5Q' in df.columns:
        df['C5_minus_L5B'] = df['C5Q'] - df['L5Q'] * wavelength_L5
        df['C5Q_noise'] = df['C5_minus_L5B'].diff()
        df.loc[df.index[0], 'C5Q_noise'] = 0

    return df


def process_single_file(file_path):
    """处理单个文件并返回处理后的数据"""
    try:
        df = pd.read_csv(file_path, delimiter=',')
        df.columns = df.columns.str.strip()

        # 找到时间列和高度角列
        time_column = next((col for col in df.columns if 'time' in col.lower()), None)
        el_column = next((col for col in df.columns if 'el' in col.lower()), None)

        if not time_column or not el_column:
            print(f"警告: {file_path} 缺少必要的列")
            return None

        # 转换时间格式并生成相对时间序列
        df['datetime'] = pd.to_datetime(df[time_column])
        # 计算相对于文件起始时间的分钟数
        df['minutes_from_start'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds() / 60
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

        df = df.rename(columns=column_mapping)
        for col in column_mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')

        l1_group = ['L1C', 'C1C']
        l5_group = ['L5Q', 'C5Q']

        l1_exists = all(col in df.columns for col in l1_group)
        l5_exists = all(col in df.columns for col in l5_group)

        if not l1_exists and not l5_exists:
            print(f"警告: {file_path} 没有可用的观测值组")
            return None

        # 数据有效性检查
        if l1_exists:
            l1_valid = (df[l1_group] != 0).all(axis=1) & (~df[l1_group].isna().any(axis=1))
            if not l1_valid.all():
                df.loc[~l1_valid, l1_group] = np.nan

        if l5_exists:
            l5_valid = (df[l5_group] != 0).all(axis=1) & (~df[l5_group].isna().any(axis=1))
            if not l5_valid.all():
                df.loc[~l5_valid, l5_group] = np.nan

        df = calculate_pseudo_noise(df)
        df['satellite'] = Path(file_path).stem

        return df

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")
        return None


def set_plot_style(axis_font_size=12, tick_font_size=10, legend_font_size=10):
    """设置全局绘图样式，允许自定义字体大小"""
    # 创建字体属性对象
    chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 宋体
    english_font = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf')  # Times New Roman

    # 设置字体大小属性
    chinese_font.set_size(axis_font_size)
    english_font.set_size(axis_font_size)

    # 创建额外的字体对象用于刻度和图例
    tick_chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
    tick_english_font = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf')
    tick_chinese_font.set_size(tick_font_size)
    tick_english_font.set_size(tick_font_size)

    legend_chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
    legend_english_font = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf')
    legend_chinese_font.set_size(legend_font_size)
    legend_english_font.set_size(legend_font_size)

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'custom'

    return {
        'axis': {'chinese': chinese_font, 'english': english_font},
        'tick': {'chinese': tick_chinese_font, 'english': tick_english_font},
        'legend': {'chinese': legend_chinese_font, 'english': legend_english_font}
    }


def create_mixed_label(chinese_text, english_text, font):
    """创建混合中英文的标签"""
    return f'{chinese_text} {english_text}'


def plot_pseudo_noise_histogram(data_frames, title=None, axis_font_size=12, tick_font_size=10, legend_font_size=10):
    """绘制多卫星伪距噪声分布直方图"""
    fonts = set_plot_style(axis_font_size, tick_font_size, legend_font_size)
    fig, ax = plt.subplots(figsize=(10, 6.18))

    c1_noise = []
    c5_noise = []

    # 数据收集部分保持不变
    for sat_name, df in data_frames.items():
        if 'C1C_noise' in df.columns:
            valid_mask = (~df['C1C_noise'].isna()) & (df.index > 0)
            if valid_mask.any():
                c1_noise.extend(df.loc[valid_mask, 'C1C_noise'].values)

        if 'C5Q_noise' in df.columns:
            valid_mask = (~df['C5Q_noise'].isna()) & (df.index > 0)
            if valid_mask.any():
                c5_noise.extend(df.loc[valid_mask, 'C5Q_noise'].values)

    rms_c1 = np.sqrt(np.mean(np.array(c1_noise) ** 2)) if c1_noise else 0
    rms_c5 = np.sqrt(np.mean(np.array(c5_noise) ** 2)) if c5_noise else 0

    # 修改bins的数量和计算方式
    bins = np.linspace(-4, 4, 200)  # 减少bins数量
    bin_width = bins[1] - bins[0]  # 计算bin宽度

    # 修改直方图绘制方式
    if c1_noise:
        # 计算每个bin的频率并转换为百分比
        hist, _ = np.histogram(c1_noise, bins=bins)
        percentage = hist / len(c1_noise) * 100
        ax.bar(bins[:-1], percentage, bin_width,
               color='blue', alpha=0.6, label='B1I', edgecolor='blue')

    if c5_noise:
        hist, _ = np.histogram(c5_noise, bins=bins)
        percentage = hist / len(c5_noise) * 100
        ax.bar(bins[:-1], percentage, bin_width,
               color='red', alpha=0.6, label='E5a', edgecolor='red')

    # RMS值显示 - 使用自定义字体大小
    plt.figtext(0.15, 0.85, f'RMS: {rms_c1:.3f}m', color='blue',
                fontproperties=fonts['tick']['english'], fontsize=tick_font_size)
    plt.figtext(0.15, 0.80, f'RMS: {rms_c5:.3f}m', color='red',
                fontproperties=fonts['tick']['english'], fontsize=tick_font_size)

    # 坐标轴设置
    ax.set_xlabel(create_mixed_label('伪距噪声', '(m)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])
    ax.set_ylabel(create_mixed_label('分布率', '(%)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])

    ax.set_ylim(0, 10)  # 设置一个合理的y轴范围
    ax.set_xlim(-4, 4)
    ax.grid(True, linestyle=':', alpha=0.3)

    # 图例设置 - 使用自定义字体大小
    ax.legend(loc='upper right', fancybox=False, edgecolor='black',
              prop=fonts['legend']['english'])

    # 刻度标签设置 - 使用自定义字体大小
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fonts['tick']['english'])

    plt.tight_layout()
    plt.show()


def plot_pseudo_noise_time_multi(data_frames, title=None, axis_font_size=12, tick_font_size=10, legend_font_size=10):
    """绘制多卫星伪距噪声随时间变化的图"""
    fonts = set_plot_style(axis_font_size, tick_font_size, legend_font_size)

    fig, ax = plt.subplots(figsize=(10, 6.18))

    # 其他设置保持不变
    markers = ['o']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}
    c1_color = 'blue'
    c5_color = 'red'
    stats = {'C1': [], 'C5': []}

    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]

        # 处理C1波段数据
        if 'C1C_noise' in df.columns:
            # 创建有效数据的掩码
            valid_mask = (~df['C1C_noise'].isna()) & (~df['minutes_from_start'].isna()) & (df.index > 0)
            if valid_mask.any():
                # 使用掩码选择对应的时间和数据点
                time_points = df.loc[valid_mask, 'minutes_from_start']
                noise_points = df.loc[valid_mask, 'C1C_noise']

                ax.plot(time_points, noise_points,
                        color=c1_color, linestyle='-', linewidth=1,
                        marker=marker, markersize=1.2, alpha=0.7,
                        label='P1')

                std_c1 = noise_points.std()
                stats['C1'].append((sat_name, std_c1))
                print(f"{sat_name}-C1波段: 标准差={std_c1:.3f}m")

        # 处理C5波段数据
        if 'C5Q_noise' in df.columns:
            valid_mask = (~df['C5Q_noise'].isna()) & (~df['minutes_from_start'].isna()) & (df.index > 0)
            if valid_mask.any():
                time_points = df.loc[valid_mask, 'minutes_from_start']
                noise_points = df.loc[valid_mask, 'C5Q_noise']

                ax.plot(time_points, noise_points,
                        color=c5_color, linestyle='-', linewidth=1,
                        marker=marker, markersize=1.2, alpha=0.7, zorder=3,
                        label='P5')

                std_c5 = noise_points.std()
                stats['C5'].append((sat_name, std_c5))
                print(f"{sat_name}-C5波段: 标准差={std_c5:.3f}m")

    # 计算并显示平均标准差
    if stats['C1']:
        mean_std_c1 = np.mean([std for _, std in stats['C1']])
        print(f"\nC1波段多卫星标准差平均值: {mean_std_c1:.3f}m")

    if stats['C5']:
        mean_std_c5 = np.mean([std for _, std in stats['C5']])
        print(f"C5波段多卫星标准差平均值: {mean_std_c5:.3f}m")

    # 设置轴标签 - 使用混合字体和自定义字体大小
    ax.set_xlabel(create_mixed_label('观测时间', '(分钟)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])
    ax.set_ylabel(create_mixed_label('伪距噪声', '(m)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])

    if title:
        ax.set_title(title, fontproperties=fonts['axis']['chinese'])

    ax.set_ylim(-40, 40)
    ax.grid(True, linestyle='-', alpha=0.3)

    # 设置图例和刻度标签 - 使用自定义字体大小
    ax.legend(loc='upper right', prop=fonts['legend']['english'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fonts['tick']['english'])

    plt.tight_layout()
    plt.show()


def plot_pseudo_noise_elevation_multi(data_frames, title=None, axis_font_size=12, tick_font_size=10,
                                      legend_font_size=10):
    """绘制多卫星伪距噪声随高度角变化的图"""
    fonts = set_plot_style(axis_font_size, tick_font_size, legend_font_size)

    fig, ax = plt.subplots(figsize=(10, 6.18))

    # 其他设置保持不变
    markers = ['o']
    sat_markers = {sat: markers[i % len(markers)] for i, sat in enumerate(data_frames.keys())}
    c1_color = 'blue'
    c5_color = 'red'

    # 绘图部分保持不变
    for sat_name, df in data_frames.items():
        marker = sat_markers[sat_name]

        if 'C1C_noise' in df.columns:
            valid_mask = (~df['C1C_noise'].isna()) & (~df['el'].isna()) & (df.index > 0)
            if valid_mask.any():
                ax.scatter(df.loc[valid_mask, 'el'],
                           df.loc[valid_mask, 'C1C_noise'],
                           c=c1_color, marker=marker, s=1, alpha=0.6,
                           label=f'B1I')

        if 'C5Q_noise' in df.columns:
            valid_mask = (~df['C5Q_noise'].isna()) & (~df['el'].isna()) & (df.index > 0)
            if valid_mask.any():
                ax.scatter(df.loc[valid_mask, 'el'],
                           df.loc[valid_mask, 'C5Q_noise'],
                           c=c5_color, marker=marker, s=1, alpha=0.6,
                           label=f'E5a')

    # 设置轴标签 - 使用混合字体和自定义字体大小
    ax.set_xlabel(create_mixed_label('卫星高度角', '(°)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])
    ax.set_ylabel(create_mixed_label('伪距噪声', '(m)', fonts['axis']['chinese']),
                  fontproperties=fonts['axis']['chinese'])

    if title:
        ax.set_title(title, fontproperties=fonts['axis']['chinese'])

    ax.set_ylim(-40, 40)
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.grid(True, linestyle='-', alpha=0.4)

    # 设置图例和刻度标签 - 使用自定义字体大小
    ax.legend(loc='upper right', prop=fonts['legend']['english'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fonts['tick']['english'])

    plt.tight_layout()
    plt.show()


# 主函数修改为接受字体大小参数
def main(axis_font_size=12, tick_font_size=10, legend_font_size=10):
    """主函数 - 处理数据并生成图表"""
    folder_path = r'C:\Users\zhang\Desktop\paper\data_analysis\P40\BDS'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print("未找到CSV文件")
        return

    processed_data = {}
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"正在处理文件: {file_name}")

        df = process_single_file(file_path)
        if df is not None:
            satellite_id = Path(file_name).stem
            processed_data[satellite_id] = df

    if not processed_data:
        print("没有成功处理任何文件")
        return

    # 传递字体大小参数给各个绘图函数
    plot_pseudo_noise_histogram(processed_data, axis_font_size=axis_font_size,
                                tick_font_size=tick_font_size, legend_font_size=legend_font_size)
    # plot_pseudo_noise_time_multi(processed_data, axis_font_size=axis_font_size,
    #                            tick_font_size=tick_font_size, legend_font_size=legend_font_size)
    plot_pseudo_noise_elevation_multi(processed_data, axis_font_size=axis_font_size,
                                      tick_font_size=tick_font_size, legend_font_size=legend_font_size)


if __name__ == "__main__":
    # 可以在这里调整所有图表的字体大小
    main(axis_font_size=18, tick_font_size=18, legend_font_size=15)

    # 或者运行默认大小
    # main()