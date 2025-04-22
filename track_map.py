import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rcParams, font_manager
import folium
from io import BytesIO
import os
import time
import requests
from PIL import Image
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# 设置字体
# 中文使用宋体，英文和数字使用Times New Roman
plt.rcParams['font.family'] = ['sans-serif', 'serif']
plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
plt.rcParams['font.serif'] = ['Times New Roman']  # Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 设置不同类型文本的字体
def set_fonts(axis_font_size=18, legend_font_size=18):
    # 设置坐标轴标签字体大小
    plt.rcParams['axes.labelsize'] = axis_font_size
    plt.rcParams['axes.titlesize'] = axis_font_size

    # 设置刻度标签字体大小
    plt.rcParams['xtick.labelsize'] = axis_font_size
    plt.rcParams['ytick.labelsize'] = axis_font_size

    # 使用FontProperties对象来设置不同元素的字体
    # 检查字体文件是否存在，如果不存在使用备选方案
    try:
        font_song = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
    except:
        # 备选方案：使用系统中可用的中文字体
        font_song = font_manager.FontProperties(family='SimSun')

    try:
        font_times = font_manager.FontProperties(fname="C:/Windows/Fonts/times.ttf")
    except:
        # 备选方案：使用系统中可用的Times New Roman
        font_times = font_manager.FontProperties(family='Times New Roman')

    # 设置字体大小
    font_song.set_size(axis_font_size)
    font_times.set_size(axis_font_size)

    return font_song, font_times, legend_font_size


# 从空格分隔的txt文件中读取经纬度数据（参考文件）
def read_space_separated_file(file_path):
    lons, lats = [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("%") or not line.strip():
                continue
            # 使用空格分隔
            eachData = line.split()
            if len(eachData) >= 4:  # 确保有足够的列
                lon = float(eachData[3])  # 经度
                lat = float(eachData[2])  # 纬度
                lons.append(lon)
                lats.append(lat)
    return lons, lats


# 从逗号分隔的txt文件中读取经纬度数据（对比文件）
def read_comma_separated_file(file_path):
    lons, lats = [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("%") or not line.strip():
                continue
            # 使用逗号分隔
            eachData = line.split(',')
            if len(eachData) >= 4:  # 确保有足够的列
                lon = float(eachData[3])  # 经度
                lat = float(eachData[2])  # 纬度
                lons.append(lon)
                lats.append(lat)
    return lons, lats


# 使用Cartopy绘制地图轨迹（更快的替代方案）
def plot_with_cartopy(base_lons, base_lats, rtk_lons, rtk_lats, font_song, font_times):
    print("正在绘制地图轨迹...")
    # 计算中心点
    center_lat = (np.mean(base_lats) + np.mean(rtk_lats)) / 2
    center_lon = (np.mean(base_lons) + np.mean(rtk_lons)) / 2

    # 计算区域范围（加上一定的边距）
    min_lon = min(min(base_lons), min(rtk_lons)) - 0.001
    max_lon = max(max(base_lons), max(rtk_lons)) + 0.001
    min_lat = min(min(base_lats), min(rtk_lats)) - 0.001
    max_lat = max(max(base_lats), max(rtk_lats)) + 0.001

    # 创建地图
    fig = plt.figure(figsize=(10, 10))

    # 使用OpenStreetMap作为底图（比卫星图像加载快）
    # 可以选择不同的瓦片源:
    # - cimgt.OSM() - OpenStreetMap
    # - cimgt.GoogleTiles() - Google地图
    # - cimgt.Stamen('terrain') - Stamen地形图
    tile_source = cimgt.OSM()

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.add_image(tile_source, 14)  # 14是缩放级别，可以根据需要调整

    # 设置地图范围
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # 绘制轨迹
    ax.plot(base_lons, base_lats, 'b.', markersize=5, transform=ccrs.PlateCarree())
    ax.plot(rtk_lons, rtk_lats, 'g.', markersize=5, transform=ccrs.PlateCarree())

    # 移除坐标轴标签
    ax.set_axis_off()

    # 保存图像
    plt.savefig('地图轨迹.png', dpi=300, bbox_inches='tight')
    plt.show()


# 使用谷歌卫星地图作为底图的folium地图
def create_simple_folium_map(base_lons, base_lats, rtk_lons, rtk_lats):
    print("正在生成交互式谷歌卫星地图...")
    # 计算中心点
    center_lat = (np.mean(base_lats) + np.mean(rtk_lats)) / 2
    center_lon = (np.mean(base_lons) + np.mean(rtk_lons)) / 2

    # 创建基础地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles=None,  # 不使用默认底图
        control_scale=True  # 添加比例尺
    )

    # 添加谷歌卫星地图作为底图
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # 添加BASE轨迹
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in zip(base_lats, base_lons)],
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)

    # 添加RTK轨迹
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in zip(rtk_lats, rtk_lons)],
        color='green',
        weight=3,
        opacity=0.8
    ).add_to(m)

    # 保存到HTML文件
    m.save('谷歌卫星地图轨迹.html')
    print("交互式地图已保存为 '谷歌卫星地图轨迹.html'")
    return m


if __name__ == '__main__':
    # 获取用户自定义的字体大小，如果没有指定则使用默认值18
    try:
        axis_font_size = int(input("请输入坐标轴字体大小 (默认为18): ") or 18)
        legend_font_size = int(input("请输入图例字体大小 (默认为18): ") or 18)
    except ValueError:
        print("输入无效，使用默认值18")
        axis_font_size = 18
        legend_font_size = 18

    # 设置字体
    font_song, font_times, legend_font_size = set_fonts(axis_font_size, legend_font_size)

    # 文件路径
    reference_file = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\Huace_rtklib.txt"
    comparison_file = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\p40_rp_rkf_dynamic.txt"

    # 读取参考文件数据（BASE）- 使用空格分隔
    base_lons, base_lats = read_space_separated_file(reference_file)

    # 读取第二个文件数据（RTK）- 使用逗号分隔
    rtk_lons, rtk_lats = read_comma_separated_file(comparison_file)

    # 检查数据是否成功读取
    if not base_lons or not rtk_lons:
        print("警告：数据读取可能出现问题，请检查文件格式和路径")

    # 选择地图类型
    print("地图类型选择:")
    print("1: 静态地图轨迹 (使用OpenStreetMap，加载更快)")
    print("2: 交互式地图轨迹 (使用OpenStreetMap，加载更快)")
    print("3: 只绘制站心坐标系图")
    map_type = input("请输入选择 (1/2/3): ")

    if map_type == '1':
        # 使用Cartopy绘制地图轨迹
        start_time = time.time()
        plot_with_cartopy(base_lons, base_lats, rtk_lons, rtk_lats, font_song, font_times)
        print(f"地图绘制完成，耗时 {time.time() - start_time:.2f} 秒")

    elif map_type == '2':
        # 使用简化版folium地图
        start_time = time.time()
        create_simple_folium_map(base_lons, base_lats, rtk_lons, rtk_lats)
        print(f"地图生成完成，耗时 {time.time() - start_time:.2f} 秒")

    # 无论选择什么地图类型，都绘制站心坐标系图
    if map_type in ['1', '2', '3']:
        print("正在绘制站心坐标系图...")
        start_time = time.time()

        # 选择参考点（使用BASE的第一个点）
        ref_lon = base_lons[0] if base_lons else 0
        ref_lat = base_lats[0] if base_lats else 0

        # 地球半径（米）
        R = 6378137.0

        # 经纬度转换为站心坐标系
        # 计算BASE的站心坐标
        base_e = []
        base_n = []
        for lon, lat in zip(base_lons, base_lats):
            # 经度差转换为东向距离
            e = R * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
            # 纬度差转换为北向距离
            n = R * math.radians(lat - ref_lat)
            base_e.append(e)
            base_n.append(n)

        # 计算RTK的站心坐标
        rtk_e = []
        rtk_n = []
        for lon, lat in zip(rtk_lons, rtk_lats):
            # 经度差转换为东向距离
            e = R * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
            # 纬度差转换为北向距离
            n = R * math.radians(lat - ref_lat)
            rtk_e.append(e)
            rtk_n.append(n)

        # 单独的站心坐标系图
        plt.figure(figsize=(10, 8))
        plt.scatter(base_e, base_n, s=5, c='blue', marker='.', label='BASE')
        plt.scatter(rtk_e, rtk_n, s=5, c='red', marker='.', label='P40_RKF')

        plt.xlabel('E(m)', fontproperties=font_song, fontsize=axis_font_size)
        plt.ylabel('N(m)', fontproperties=font_song, fontsize=axis_font_size)
        # plt.title('站心坐标系轨迹', fontproperties=font_song, fontsize=14)

        # 为坐标轴数字设置Times New Roman字体
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontproperties(font_times)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(prop=font_times, fontsize=legend_font_size)
        plt.axis('equal')  # 强制等比例
        plt.tight_layout()
        plt.savefig('站心坐标系轨迹.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"站心坐标系图绘制完成，耗时 {time.time() - start_time:.2f} 秒")
    else:
        print("无效的选择，请输入1, 2或3")