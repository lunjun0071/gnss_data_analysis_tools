import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np


def set_plot_style(font_size=10):
    """
    设置全局绘图样式

    参数:
    font_size (int): 默认字体大小
    """
    chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 宋体
    english_font = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf')  # Times New Roman

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'custom'

    return chinese_font, english_font


def create_mixed_label(chinese_text, english_text, chinese_font):
    """创建混合中英文的标签"""
    return f'{chinese_text} {english_text}'


def clean_data(df):
    # 清理列名
    df.columns = df.columns.str.strip()

    # 对所有列进行清理
    for column in df.columns:
        if df[column].dtype == 'object':  # 只处理字符串类型的列
            df[column] = df[column].str.strip()

    return df


def calculate_statistics(df):
    # 计算S1C和S5Q的平均值
    s1c_mean = df['S1C'].mean()
    s5q_mean = df['S5Q'].mean()

    # 打印统计结果，保留一位小数
    print("\nSignal Strength Statistics:")
    print(f"S1C Average: {s1c_mean:.1f} dB-Hz")
    print(f"S5Q Average: {s5q_mean:.1f} dB-Hz")

    return s1c_mean, s5q_mean


def plot_gnss_data(csv_file, axis_font_size=10, legend_font_size=10):
    """
    绘制GNSS信号数据

    参数:
    csv_file (str): CSV文件路径
    axis_font_size (int): 坐标轴字体大小
    legend_font_size (int): 图例字体大小
    """
    # 获取字体
    chinese_font, english_font = set_plot_style(font_size=axis_font_size)

    # 读取CSV数据
    df = pd.read_csv(csv_file)
    df = clean_data(df)

    # 将时间列转换为datetime格式
    df['datetime'] = pd.to_datetime(df['time'])

    # 转换数值列为float类型
    df['S1C'] = pd.to_numeric(df['S1C'], errors='coerce')
    df['S5Q'] = pd.to_numeric(df['S5Q'], errors='coerce')
    df['el'] = pd.to_numeric(df['el'], errors='coerce')

    # 计算统计信息
    s1c_mean, s5q_mean = calculate_statistics(df)

    # 创建图表和双Y轴
    fig, ax1 = plt.subplots(figsize=(10, 6.18))
    ax2 = ax1.twinx()

    # 绘制S1C和S5Q数据（左Y轴）
    ax1.plot(df['datetime'], df['S1C'], 'b-', label='L1', linewidth=1)
    ax1.plot(df['datetime'], df['S5Q'], 'r-', label='L5', linewidth=1)

    # 绘制高度角数据（右Y轴）
    ax2.plot(df['datetime'], df['el'], 'g-', label='Elevation', linewidth=1)

    # 设置左Y轴标签、范围和刻度
    ax1.set_ylabel(create_mixed_label('载噪比', '(dB-Hz)', chinese_font),
                   fontproperties=chinese_font, fontsize=axis_font_size)
    ax1.set_ylim(0, 60)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # 设置刻度标签字体和大小
    for label in ax1.get_yticklabels():
        label.set_fontproperties(english_font)
        label.set_fontsize(axis_font_size)

    # 设置右Y轴标签、范围和刻度
    ax2.set_ylabel(create_mixed_label('高度角', '(°)', chinese_font),
                   fontproperties=chinese_font, color='g', fontsize=axis_font_size)
    ax2.set_ylim(0, 90)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # 设置右Y轴刻度标签字体和大小
    for label in ax2.get_yticklabels():
        label.set_fontproperties(english_font)
        label.set_fontsize(axis_font_size)
        label.set_color('g')

    # 设置X轴格式和标签
    ax1.set_xlabel(create_mixed_label('UTC', '(hh:mm)', chinese_font),
                   fontproperties=chinese_font, fontsize=axis_font_size)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 设置X轴刻度标签字体和大小
    for label in ax1.get_xticklabels():
        label.set_fontproperties(english_font)
        label.set_fontsize(axis_font_size)

    plt.gcf().autofmt_xdate()

    # 设置网格
    ax1.grid(True, alpha=0.3)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                        loc='upper right', prop=english_font)

    # 设置图例字体大小
    for text in legend.get_texts():
        text.set_fontsize(legend_font_size)

    # 设置标题
    satellite_id = df['> PRN'].iloc[0].strip()
    # plt.title(f'Huace M11Pro {satellite_id}', fontproperties=english_font, fontsize=axis_font_size + 2)

    # 调整布局
    plt.tight_layout()

    return fig


def plot_cn0_histogram(df, title=None, axis_font_size=10, legend_font_size=10):
    """
    绘制C/N0分布直方图

    参数:
    df (DataFrame): 包含GNSS数据的DataFrame
    title (str, optional): 图表标题
    axis_font_size (int): 坐标轴字体大小
    legend_font_size (int): 图例字体大小
    """
    chinese_font, english_font = set_plot_style(font_size=axis_font_size)

    fig, ax = plt.subplots(figsize=(10, 6.18))

    # 收集数据
    s1c_data = df['S1C'].dropna().values
    s5q_data = df['S5Q'].dropna().values

    # 计算RMS
    rms_s1c = np.sqrt(np.mean(s1c_data ** 2)) if len(s1c_data) > 0 else 0
    rms_s5q = np.sqrt(np.mean(s5q_data ** 2)) if len(s5q_data) > 0 else 0

    # 设置bins和直方图参数
    bins = np.linspace(10, 60, 250)  # 修改范围从10到60
    bin_width = bins[1] - bins[0]

    # 绘制直方图 - 修改计算方式
    if len(s1c_data) > 0:
        # 计算频数并手动转换为百分比
        counts, edges = np.histogram(s1c_data, bins=bins)
        total_count = len(s1c_data)
        percentages = (counts / total_count) * 100  # 转换为百分比

        # 绘制直方图柱状图
        ax.bar(edges[:-1], percentages, bin_width,
               color='blue', alpha=0.6, label='L1', edgecolor='blue')

    if len(s5q_data) > 0:
        counts, edges = np.histogram(s5q_data, bins=bins)
        total_count = len(s5q_data)
        percentages = (counts / total_count) * 100

        ax.bar(edges[:-1], percentages, bin_width,
               color='red', alpha=0.6, label='L5', edgecolor='red')

    # 设置RMS值文本
    plt.figtext(0.15, 0.85, f'RMS: {rms_s1c:.1f} dB-Hz', color='blue',
                fontproperties=english_font, fontsize=axis_font_size)
    plt.figtext(0.15, 0.80, f'RMS: {rms_s5q:.1f} dB-Hz', color='red',
                fontproperties=english_font, fontsize=axis_font_size)

    # 设置轴标签
    ax.set_xlabel(create_mixed_label('载噪比', '(dB-Hz)', chinese_font),
                  fontproperties=chinese_font, fontsize=axis_font_size)
    ax.set_ylabel(create_mixed_label('分布率', '(%)', chinese_font),
                  fontproperties=chinese_font, fontsize=axis_font_size)

    # 设置图表属性
    ax.set_ylim(0, 15)  # 设置y轴范围为0-35%
    ax.set_xlim(10, 60)  # 设置x轴范围为35-60 dB-Hz
    ax.grid(True, linestyle=':', alpha=0.3)

    # 设置图例
    legend = ax.legend(loc='upper right', fancybox=False, edgecolor='black',
                       prop=english_font)

    # 设置图例字体大小
    for text in legend.get_texts():
        text.set_fontsize(legend_font_size)

    # 设置刻度标签字体和大小
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(english_font)
        label.set_fontsize(axis_font_size)

    if title:
        ax.set_title(title, fontproperties=chinese_font, fontsize=axis_font_size + 2)

    plt.tight_layout()
    return fig


# 使用示例
if __name__ == "__main__":
    # 用户可以设置字体大小参数
    AXIS_FONT_SIZE = 18  # 坐标轴字体大小
    LEGEND_FONT_SIZE = 15  # 图例字体大小

    file_path = 'C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\oneplusresult\\G14.csv'
    df = pd.read_csv(file_path)
    df = clean_data(df)

    # 转换数值列
    df['S1C'] = pd.to_numeric(df['S1C'], errors='coerce')
    df['S5Q'] = pd.to_numeric(df['S5Q'], errors='coerce')

    # 绘制时间序列图
    fig1 = plot_gnss_data(file_path,
                          axis_font_size=AXIS_FONT_SIZE,
                          legend_font_size=LEGEND_FONT_SIZE)
    plt.figure(1)
    plt.show()

    # 绘制直方图
    satellite_id = df['> PRN'].iloc[0].strip()
    fig2 = plot_cn0_histogram(df,
                              axis_font_size=AXIS_FONT_SIZE,
                              legend_font_size=LEGEND_FONT_SIZE)
    plt.show()
    #'C:\\Users\\zhang\\Desktop\\pythoncode\\ringo-v0.9.2-win64\\'