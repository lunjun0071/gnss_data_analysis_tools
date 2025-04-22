import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np  # 添加numpy用于统计计算

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_satellite_count():
    # 创建图形并设置比例
    fig = plt.figure(figsize=(10, 6))

    # 读取数据
    times = []
    nsat = []
    with open('C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\oneplusresult\\oneplus12_satnum_static.txt', 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            data = line.split()
            time_str = data[0] + ' ' + data[1]
            time = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S.%f')
            sat_num = int(data[2])

            times.append(time)
            nsat.append(sat_num)

    # 计算统计值
    mean_sats = np.mean(nsat)
    max_sats = np.max(nsat)
    min_sats = np.min(nsat)
    std_sats = np.std(nsat)

    # 输出统计结果
    print(f"\n卫星数量统计信息:")
    print(f"平均卫星数量: {mean_sats:.2f}")
    print(f"最大卫星数量: {max_sats}")
    print(f"最小卫星数量: {min_sats}")
    print(f"卫星数量标准差: {std_sats:.2f}")

    # 创建图表
    ax = plt.gca()
    ax.plot(times, nsat, '-o', linewidth=1, markersize=2, label='卫星数量')

    # 添加平均值线
    ax.axhline(y=mean_sats, color='r', linestyle='--', label=f'平均值 ({mean_sats:.2f})')

    # 设置时间轴格式，只显示小时和分钟
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 设置y轴为整数刻度
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 设置图像比例
    ax.set_aspect('0.003')

    # 设置标题和标签
    ax.set_title('一加12')
    ax.set_xlabel('时间 (GPST)')
    ax.set_ylabel('卫星数量')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)

    # 添加图例
    ax.legend()

    # 调整y轴范围为整数
    ymin = int(min(nsat) - 1)
    ymax = int(max(nsat) + 1)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_satellite_count()

