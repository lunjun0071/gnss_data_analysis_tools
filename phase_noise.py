import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse


def calculate_carrier_phase_noise(df, bands=None, filter_settings=None):
    """计算载波相位噪声 (CN_L)

    公式: CN_L_i = (1/√20) * (φ_i+3 - 3φ_i+2 + 3φ_i+1 - φ_i)
    其中，下标 i 为历元数，φ 表示载波相位观测值

    参数:
        df: 包含观测数据的DataFrame
        bands: 需要计算噪声的波段列表，如['L1C', 'L5Q']
        filter_settings: 过滤设置字典
    """
    # 默认过滤设置
    default_settings = {
        'cycle_slip_threshold': 0.5,
        'noise_cycle_threshold': 10.0
    }

    # 更新设置
    if filter_settings:
        default_settings.update(filter_settings)

    # 获取过滤阈值
    cycle_slip_threshold = default_settings['cycle_slip_threshold']
    noise_cycle_threshold = default_settings['noise_cycle_threshold']

    # 如果未指定波段，默认计算所有可用波段
    available_bands = [col for col in ['L1C', 'L5Q'] if col in df.columns]
    if bands is None:
        bands = available_bands
    else:
        # 确保所有指定的波段都存在于数据中
        bands = [band for band in bands if band in available_bands]
        if not bands:
            raise ValueError(f"指定的波段 {bands} 在数据中不存在。可用波段: {available_bands}")

    # 检查每个波段的数据质量
    for band in bands:
        # 检查非有限值
        invalid_mask = ~np.isfinite(df[band])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            print(f"警告: {band}波段包含 {invalid_count} 个非有限值，这些值将被标记为NaN")
            df.loc[invalid_mask, band] = np.nan

        # 检查数据中的周跳变（大于阈值的跳变）
        phase_diffs = df[band].diff().abs()
        cycle_slips = phase_diffs > cycle_slip_threshold
        cycle_slip_count = cycle_slips.sum()

        if cycle_slip_count > 0:
            print(
                f"警告: {band}波段检测到 {cycle_slip_count} 个可能的周跳变（>{cycle_slip_threshold}周），这可能影响噪声计算")

    # 为每个波段创建噪声列
    for band in bands:
        noise_col = f"{band}_noise"
        df[noise_col] = np.nan

        # 需要至少4个连续历元才能计算
        for i in range(len(df) - 3):
            # 确保所有4个点都是有效值
            if np.isfinite(df.loc[df.index[i], band]) and \
                    np.isfinite(df.loc[df.index[i + 1], band]) and \
                    np.isfinite(df.loc[df.index[i + 2], band]) and \
                    np.isfinite(df.loc[df.index[i + 3], band]):

                df.loc[df.index[i], noise_col] = (1 / np.sqrt(20)) * (
                        df.loc[df.index[i + 3], band] -
                        3 * df.loc[df.index[i + 2], band] +
                        3 * df.loc[df.index[i + 1], band] -
                        df.loc[df.index[i], band]
                )

                # 检查计算结果是否合理（例如是否太大）
                if abs(df.loc[df.index[i], noise_col]) > noise_cycle_threshold:
                    print(f"警告: 在索引 {i} 处计算的 {band} 噪声值异常大: {df.loc[df.index[i], noise_col]:.4f}周")
                    df.loc[df.index[i], noise_col] = np.nan

    return df


def calculate_pseudorange_noise(df, bands=None, filter_settings=None):
    """计算伪距噪声

    公式: CN_L_i = (1/√20) * (C_i+3 - 3C_i+2 + 3C_i+1 - C_i)
    其中，下标 i 为历元数，C 表示伪距观测值

    参数:
        df: 包含观测数据的DataFrame
        bands: 需要计算噪声的波段列表，如['C1C', 'C5Q']
        filter_settings: 过滤设置字典
    """
    # 默认过滤设置
    default_settings = {
        'pr_jump_threshold': 10.0,  # 伪距跳变阈值（米）
        'noise_meter_threshold': 100.0  # 噪声阈值（米）
    }

    # 更新设置
    if filter_settings:
        default_settings.update(filter_settings)

    # 获取过滤阈值
    pr_jump_threshold = default_settings['pr_jump_threshold']
    noise_meter_threshold = default_settings['noise_meter_threshold']

    # 如果未指定波段，默认计算所有可用波段
    available_bands = [col for col in ['C1C', 'C5Q'] if col in df.columns]
    if bands is None:
        bands = available_bands
    else:
        # 确保所有指定的波段都存在于数据中
        bands = [band for band in bands if band in available_bands]
        if not bands:
            raise ValueError(f"指定的波段 {bands} 在数据中不存在。可用波段: {available_bands}")

    # 检查每个波段的数据质量
    for band in bands:
        # 检查非有限值
        invalid_mask = ~np.isfinite(df[band])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            print(f"警告: {band}波段包含 {invalid_count} 个非有限值，这些值将被标记为NaN")
            df.loc[invalid_mask, band] = np.nan

        # 检查数据中的异常跳变
        range_diffs = df[band].diff().abs()
        jumps = range_diffs > pr_jump_threshold
        jump_count = jumps.sum()

        if jump_count > 0:
            print(f"警告: {band}波段检测到 {jump_count} 个异常跳变（>{pr_jump_threshold}米），这可能影响噪声计算")

    # 为每个波段创建噪声列
    for band in bands:
        noise_col = f"{band}_noise"
        df[noise_col] = np.nan

        # 需要至少4个连续历元才能计算
        for i in range(len(df) - 3):
            # 确保所有4个点都是有效值
            if np.isfinite(df.loc[df.index[i], band]) and \
                    np.isfinite(df.loc[df.index[i + 1], band]) and \
                    np.isfinite(df.loc[df.index[i + 2], band]) and \
                    np.isfinite(df.loc[df.index[i + 3], band]):

                df.loc[df.index[i], noise_col] = (1 / np.sqrt(20)) * (
                        df.loc[df.index[i + 3], band] -
                        3 * df.loc[df.index[i + 2], band] +
                        3 * df.loc[df.index[i + 1], band] -
                        df.loc[df.index[i], band]
                )

                # 检查计算结果是否合理
                if abs(df.loc[df.index[i], noise_col]) > noise_meter_threshold:
                    print(f"警告: 在索引 {i} 处计算的 {band} 噪声值异常大: {df.loc[df.index[i], noise_col]:.4f}米")
                    df.loc[df.index[i], noise_col] = np.nan

    return df


def plot_noise(df, bands=None, obs_type='carrier_phase', filter_settings=None):
    """绘制GNSS观测噪声二维网格图（支持中文显示）

    参数:
        df: 包含观测数据的DataFrame
        bands: 需要绘制噪声的波段列表
        obs_type: 观测类型，'carrier_phase'或'pseudorange'
        filter_settings: 过滤设置字典
    """
    # 默认过滤设置
    default_settings = {
        'std_outlier_factor': 5.0,
        'iqr_outlier_factor': 1.5,
        'data_range_threshold': 100.0,
        'meter_noise_threshold': 1.0
    }

    # 更新设置
    if filter_settings:
        default_settings.update(filter_settings)

    # 获取过滤阈值
    std_factor = default_settings['std_outlier_factor']
    iqr_factor = default_settings['iqr_outlier_factor']
    range_threshold = default_settings['data_range_threshold']
    meter_threshold = default_settings['meter_noise_threshold']

    # 输出当前使用的过滤设置
    print("\n使用的过滤设置:")
    print(f"标准差异常值倍数: {std_factor}")
    print(f"IQR异常值倍数: {iqr_factor}")
    print(f"数据范围阈值: {range_threshold}{'周' if obs_type == 'carrier_phase' else '米'}")
    print(f"噪声阈值: {meter_threshold}{'米' if obs_type == 'carrier_phase' else '米'}")

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 确定要绘制的波段和观测类型
    if obs_type == 'carrier_phase':
        available_noise_bands = [col for col in df.columns if
                                 col.endswith('_noise') and col.startswith(('L1', 'L2', 'L5'))]
        unit = '米'
        title_prefix = '载波相位'
        formula = '(CN_L = (1/√20)*(φ_i+3 - 3φ_i+2 + 3φ_i+1 - φ_i))'
    else:  # pseudorange
        available_noise_bands = [col for col in df.columns if
                                 col.endswith('_noise') and col.startswith(('C1', 'C2', 'C5'))]
        unit = '米'
        title_prefix = '伪距'
        formula = '(CN_C = (1/√20)*(C_i+3 - 3C_i+2 + 3C_i+1 - C_i))'

    if not available_noise_bands:
        raise ValueError(f"数据中未发现{title_prefix}噪声列。请先计算噪声。")

    if bands is not None:
        # 转换波段名称为噪声列名称
        noise_bands = [f"{band}_noise" for band in bands]
        # 筛选出有效的噪声列
        noise_bands = [band for band in noise_bands if band in available_noise_bands]
        if not noise_bands:
            raise ValueError(f"指定的波段 {bands} 未计算噪声。可用噪声列: {available_noise_bands}")
    else:
        noise_bands = available_noise_bands

    # 创建图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 移除NaN值
    valid_data = df.dropna(subset=noise_bands)

    # 波长字典 - 用于将周转换为米（仅载波相位需要）
    wavelengths = {
        'L1C_noise': 299792458 / 1575.42e6,  # GPS L1信号波长，约0.19米/周
        'L5Q_noise': 299792458 / 1176.45e6,  # GPS L5信号波长，约0.25米/周
    }

    # 如果是载波相位，显示波长信息
    if obs_type == 'carrier_phase':
        print("\n波长信息:")
        for band, wavelength in wavelengths.items():
            print(f"{band.split('_')[0]}波长: {wavelength:.4f} 米/周")

    # 颜色和标记样式，统一使用圆点
    styles = {
        'L1C_noise': {'color': 'b', 'marker': 'o', 'label': 'L1波段'},
        'L5Q_noise': {'color': 'r', 'marker': 'o', 'label': 'L5波段'},
        'C1C_noise': {'color': 'g', 'marker': 'o', 'label': 'C1波段'},
        'C5Q_noise': {'color': 'm', 'marker': 'o', 'label': 'C5波段'}
    }

    # 存储噪声数据（米）
    noise_data_meters = {}

    # 绘制每个波段的噪声
    for band in noise_bands:
        base_band = band.split('_')[0]  # 获取基础波段名称（不含_noise）
        noise_data = valid_data[band]

        # 对于载波相位，需要将周转换为米
        if obs_type == 'carrier_phase' and band in wavelengths:
            wavelength = wavelengths[band]
            # 转换前检查数据范围，判断是否需要检查异常值
            data_range = noise_data.max() - noise_data.min()

            if data_range > range_threshold:  # 使用设置的数据范围阈值
                print(f"警告: {base_band}波段噪声范围异常大: {data_range:.2f}周。将尝试过滤异常值。")
                # 使用中位数和MAD（中位数绝对偏差）来过滤异常值，这比均值和标准差更稳健
                median = noise_data.median()
                mad = (noise_data - median).abs().median() * 1.4826  # 常数因子使MAD等价于高斯分布的标准差
                lower_bound = median - std_factor * mad  # 使用设置的标准差倍数
                upper_bound = median + std_factor * mad

                # 标记异常值
                outlier_mask = (noise_data < lower_bound) | (noise_data > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    print(f"检测到 {outlier_count} 个异常值，将被替换为NaN")
                    # 创建一个副本，将异常值替换为NaN
                    filtered_noise = noise_data.copy()
                    filtered_noise[outlier_mask] = np.nan
                    noise_data_meters[band] = filtered_noise * wavelength
                else:
                    noise_data_meters[band] = noise_data * wavelength
            else:
                # 数据范围合理，直接转换
                noise_data_meters[band] = noise_data * wavelength

            # 显示转换后的数据范围
            valid_mask = np.isfinite(noise_data_meters[band])
            if valid_mask.any():
                min_val = noise_data_meters[band][valid_mask].min()
                max_val = noise_data_meters[band][valid_mask].max()
                print(f"{base_band}波段噪声范围(米): {min_val:.8f}m 至 {max_val:.8f}m")
            else:
                print(f"{base_band}波段没有有效数据")
        else:
            # 伪距观测值已经是米单位，直接使用
            noise_data_meters[band] = noise_data

            # 对伪距噪声也进行数据验证
            data_range = noise_data.max() - noise_data.min()
            if data_range > range_threshold:
                print(f"警告: {base_band}波段噪声范围异常大: {data_range:.2f}米。将尝试过滤异常值。")
                # 使用中位数和MAD过滤
                median = noise_data.median()
                mad = (noise_data - median).abs().median() * 1.4826
                lower_bound = median - std_factor * mad
                upper_bound = median + std_factor * mad

                outlier_mask = (noise_data < lower_bound) | (noise_data > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    print(f"检测到 {outlier_count} 个异常值，将被替换为NaN")
                    filtered_noise = noise_data.copy()
                    filtered_noise[outlier_mask] = np.nan
                    noise_data_meters[band] = filtered_noise

            valid_mask = np.isfinite(noise_data_meters[band])
            if valid_mask.any():
                min_val = noise_data_meters[band][valid_mask].min()
                max_val = noise_data_meters[band][valid_mask].max()
                print(f"{base_band}波段噪声范围: {min_val:.4f}m 至 {max_val:.4f}m")
            else:
                print(f"{base_band}波段没有有效数据")

        # 绘制噪声图前检查数据有效性
        valid_mask = np.isfinite(noise_data_meters[band])
        if valid_mask.any():
            valid_noise_data = noise_data_meters[band][valid_mask]

            # 检查数据量级，设置合理的y轴范围
            data_max = np.abs(valid_noise_data).max()

            # 如果数据可能有量级问题，尝试进一步筛选
            if data_max > meter_threshold:  # 使用设置的米单位阈值
                print(f"警告: {base_band}波段噪声数据最大值为 {data_max:.4f}m，超过阈值 {meter_threshold}m")
                print(f"尝试使用IQR倍数 {iqr_factor} 进一步过滤异常值...")

                # 使用设置的IQR倍数
                q1 = np.percentile(valid_noise_data, 25)
                q3 = np.percentile(valid_noise_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr

                normal_range_mask = (valid_noise_data >= lower_bound) & (valid_noise_data <= upper_bound)
                filtered_data = valid_noise_data[normal_range_mask]

                if len(filtered_data) > 0:
                    print(f"过滤后数据范围: {filtered_data.min():.8f}m 至 {filtered_data.max():.8f}m")

                    # 使用过滤后的数据绘图（改为散点图）
                    if band in styles:
                        ax.scatter(np.arange(len(filtered_data)), filtered_data,
                                   color=styles[band]['color'],
                                   marker='o',  # 统一使用圆点
                                   s=25,
                                   alpha=0.7,
                                   label=f"{styles[band]['label']} (过滤后)")
                    else:
                        # 默认样式
                        ax.scatter(np.arange(len(filtered_data)), filtered_data,
                                   marker='o',  # 统一使用圆点
                                   s=25,
                                   alpha=0.7,
                                   label=f"{base_band}波段 (过滤后)")

                    # 更新存储的数据以便计算统计值
                    temp_series = pd.Series(filtered_data)
                    noise_data_meters[band] = temp_series
                else:
                    print(f"过滤后没有剩余有效数据，将跳过 {base_band} 波段的绘制")
                    # 创建空Series以避免后续计算错误
                    noise_data_meters[band] = pd.Series([])
            else:
                # 数据量级合理，使用散点图而非折线图绘制
                if band in styles:
                    ax.scatter(np.arange(len(valid_noise_data)), valid_noise_data,
                               color=styles[band]['color'],
                               marker='o',  # 统一使用圆点
                               s=25,  # 点的大小
                               alpha=0.7,  # 透明度
                               label=styles[band]['label'])
                else:
                    # 默认样式
                    ax.scatter(np.arange(len(valid_noise_data)), valid_noise_data,
                               marker='o',  # 统一使用圆点
                               s=25,
                               alpha=0.7,
                               label=f"{base_band}波段")
        else:
            print(f"{base_band}波段没有有效数据，跳过绘制")

    # 设置图表属性
    ax.set_xlabel('观测点序号', fontsize=12)
    ax.set_ylabel(f'{title_prefix}噪声 ({unit})', fontsize=12)
    title = f'GNSS{title_prefix}观测噪声'
    if len(noise_bands) == 1:
        title = f"{noise_bands[0].split('_')[0]}波段 - {title}"
    ax.set_title(f'{title} {formula}', fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.legend(loc='upper right')

    # 计算适当的x轴刻度间隔
    num_points = len(valid_data)

    # 防止除零错误
    if num_points == 0:
        # 如果没有有效数据，不设置刻度
        xticks = []
    else:
        # 确保tick_interval至少为1
        if num_points > 1000:
            tick_interval = max(1, num_points // 10)
        else:
            tick_interval = max(1, num_points // 5)

        # 只在重要位置显示x轴刻度
        xticks = np.arange(0, num_points, tick_interval)

    # 设置x轴刻度
    if len(xticks) > 0:  # 使用len()检查数组是否为空
        ax.set_xticks(xticks)

        # 设置刻度标签格式
        times = valid_data['datetime'] if 'datetime' in valid_data.columns else None
        if times is not None and hasattr(times, 'dt') and len(times) > 0:
            # 转换索引位置为对应的时间标签
            time_labels = [times.iloc[i].strftime('%H:%M:%S') if i < len(times) else '' for i in xticks]
            ax.set_xticklabels(time_labels, rotation=30, ha='right')

    # 设置背景为白色，网格线为灰色
    ax.set_facecolor('white')
    plt.grid(True, linestyle='-', color='gray', linewidth=0.8)

    plt.tight_layout()
    plt.show()

    # 输出统计信息
    print(f"\n{title_prefix}噪声统计信息({unit}):")
    for band in noise_bands:
        base_band = band.split('_')[0]
        noise_meters = noise_data_meters[band]

        # 数据验证和过滤
        filtered_noise = noise_meters.copy()

        # 检查非有限值（NaN或Inf）
        invalid_count = np.sum(~np.isfinite(filtered_noise))
        if invalid_count > 0:
            print(f"警告: {base_band}波段包含 {invalid_count} 个非有限值，将被过滤")
            filtered_noise = filtered_noise[np.isfinite(filtered_noise)]

        # 检查异常值（超过均值±5个标准差）
        if len(filtered_noise) > 0:
            mean = np.mean(filtered_noise)
            std = np.std(filtered_noise)
            lower_bound = mean - 5 * std
            upper_bound = mean + 5 * std
            outlier_mask = (filtered_noise < lower_bound) | (filtered_noise > upper_bound)
            outlier_count = np.sum(outlier_mask)

            if outlier_count > 0:
                print(f"警告: {base_band}波段检测到 {outlier_count} 个异常值，将被过滤")
                filtered_noise = filtered_noise[~outlier_mask]

        # 检查数据量级
        if len(filtered_noise) > 0:
            print(
                f"数据检测: {base_band}波段噪声范围: min={np.min(filtered_noise):.8f}, max={np.max(filtered_noise):.8f}")

            # 计算统计值（使用过滤后的数据）
            rms = np.sqrt(np.mean(np.square(filtered_noise)))
            std_dev = np.std(filtered_noise)
            print(f"{base_band}波段: 标准差={std_dev:.8f}{unit}, RMS={rms:.8f}{unit}")
        else:
            print(f"{base_band}波段: 过滤后无有效数据")

    # 如果是载波相位观测，则也输出周为单位的统计信息
    if obs_type == 'carrier_phase':
        print(f"\n{title_prefix}噪声统计信息(周):")
        for band in noise_bands:
            base_band = band.split('_')[0]
            # 仅对载波相位波段进行计算
            if band in wavelengths:
                noise_cycles = valid_data[band]

                # 数据验证和过滤
                filtered_cycles = noise_cycles.copy()

                # 检查非有限值
                invalid_count = np.sum(~np.isfinite(filtered_cycles))
                if invalid_count > 0:
                    print(f"警告: {base_band}波段包含 {invalid_count} 个非有限值，将被过滤")
                    filtered_cycles = filtered_cycles[np.isfinite(filtered_cycles)]

                # 检查异常值
                if len(filtered_cycles) > 0:
                    mean = np.mean(filtered_cycles)
                    std = np.std(filtered_cycles)
                    lower_bound = mean - 5 * std
                    upper_bound = mean + 5 * std
                    outlier_mask = (filtered_cycles < lower_bound) | (filtered_cycles > upper_bound)
                    outlier_count = np.sum(outlier_mask)

                    if outlier_count > 0:
                        print(f"警告: {base_band}波段检测到 {outlier_count} 个异常值，将被过滤")
                        filtered_cycles = filtered_cycles[~outlier_mask]

                # 检查数据量级
                if len(filtered_cycles) > 0:
                    print(
                        f"数据检测: {base_band}波段噪声范围(周): min={np.min(filtered_cycles):.10f}, max={np.max(filtered_cycles):.10f}")

                    # 计算统计值
                    rms_cycles = np.sqrt(np.mean(np.square(filtered_cycles)))
                    std_dev_cycles = np.std(filtered_cycles)
                    print(f"{base_band}波段: 标准差={std_dev_cycles:.10f}周, RMS={rms_cycles:.10f}周")
                else:
                    print(f"{base_band}波段: 过滤后无有效数据")


def main(input_file=None, specified_bands=None, obs_type='carrier_phase', filter_settings=None):
    """主函数 - 处理数据并生成图表

    参数:
        input_file: 可选，指定输入文件路径
        specified_bands: 可选，指定要分析的波段列表，如['L1C', 'L5Q']或['C1C', 'C5Q']
        obs_type: 可选，观测类型，'carrier_phase'或'pseudorange'
        filter_settings: 可选，过滤设置字典，包含以下键：
            - cycle_slip_threshold: 周跳变检测阈值（周）
            - noise_cycle_threshold: 噪声值阈值（周）
            - pr_jump_threshold: 伪距跳变检测阈值（米）
            - noise_meter_threshold: 伪距噪声阈值（米）
            - std_outlier_factor: 标准差异常值过滤倍数
            - iqr_outlier_factor: IQR异常值过滤倍数
            - data_range_threshold: 数据范围阈值（周或米）
            - meter_noise_threshold: 米单位噪声阈值（米）
    """
    # 设置默认的过滤参数
    default_filter_settings = {
        'cycle_slip_threshold': 5,  # 周跳变检测阈值（周）
        'noise_cycle_threshold': 55.0,  # 噪声值异常阈值（周）
        'pr_jump_threshold': 100.0,  # 伪距跳变检测阈值（米）
        'noise_meter_threshold': 100.0,  # 伪距噪声阈值（米）
        'std_outlier_factor': 50.0,  # 标准差异常值过滤倍数
        'iqr_outlier_factor': 15,  # IQR异常值过滤倍数
        'data_range_threshold': 100.0,  # 数据范围阈值（周或米）
        'meter_noise_threshold': 100.0  # 米单位噪声阈值（米）
    }

    # 如果提供了自定义过滤设置，则更新默认设置
    if filter_settings:
        default_filter_settings.update(filter_settings)

    # 存储过滤设置，以便传递给其他函数
    filter_settings = default_filter_settings

    # 如果直接通过函数参数指定了文件路径和波段，则使用这些值
    # 否则，使用命令行参数
    if input_file is None or specified_bands is None or obs_type is None:
        parser = argparse.ArgumentParser(description='GNSS观测噪声分析')
        parser.add_argument('--file', type=str, default='输入您的默认文件路径.csv',
                            help='输入CSV文件路径')
        parser.add_argument('--bands', type=str, nargs='+', default=None,
                            help='要分析的波段，例如载波相位: L1C L5Q，伪距: C1C C5Q。不指定则分析所有可用波段。')
        parser.add_argument('--type', type=str, choices=['carrier_phase', 'pseudorange'], default='carrier_phase',
                            help='观测类型: carrier_phase(载波相位) 或 pseudorange(伪距)')

        # 添加过滤设置参数
        parser.add_argument('--cycle-slip', type=float, default=default_filter_settings['cycle_slip_threshold'],
                            help='周跳变检测阈值（周），默认0.5')
        parser.add_argument('--noise-threshold', type=float, default=default_filter_settings['noise_cycle_threshold'],
                            help='载波相位噪声值异常阈值（周），默认10.0')
        parser.add_argument('--pr-jump', type=float, default=default_filter_settings['pr_jump_threshold'],
                            help='伪距跳变检测阈值（米），默认10.0')
        parser.add_argument('--pr-noise-threshold', type=float,
                            default=default_filter_settings['noise_meter_threshold'],
                            help='伪距噪声阈值（米），默认100.0')
        parser.add_argument('--std-factor', type=float, default=default_filter_settings['std_outlier_factor'],
                            help='标准差异常值过滤倍数，默认5.0')
        parser.add_argument('--iqr-factor', type=float, default=default_filter_settings['iqr_outlier_factor'],
                            help='IQR异常值过滤倍数，默认1.5')
        parser.add_argument('--range-threshold', type=float, default=default_filter_settings['data_range_threshold'],
                            help='数据范围阈值（周或米），默认100.0')
        parser.add_argument('--meter-threshold', type=float, default=default_filter_settings['meter_noise_threshold'],
                            help='米单位噪声阈值（米），默认1.0')

        args = parser.parse_args()

        file_path = input_file if input_file is not None else args.file
        bands = specified_bands if specified_bands is not None else args.bands
        obs_type = obs_type if obs_type is not None else args.type

        # 更新过滤设置
        filter_settings.update({
            'cycle_slip_threshold': args.cycle_slip,
            'noise_cycle_threshold': args.noise_threshold,
            'pr_jump_threshold': args.pr_jump,
            'noise_meter_threshold': args.pr_noise_threshold,
            'std_outlier_factor': args.std_factor,
            'iqr_outlier_factor': args.iqr_factor,
            'data_range_threshold': args.range_threshold,
            'meter_noise_threshold': args.meter_threshold
        })
    else:
        file_path = input_file
        bands = specified_bands

    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, delimiter=',')
    except Exception as e:
        print(f"读取文件失败: {e}")
        default_file = 'C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\G08.csv'
        print(f"将使用默认路径: {default_file}")
        df = pd.read_csv(default_file, delimiter=',')

    # 清理列名中的空格
    df.columns = df.columns.str.strip()

    # 找到时间列
    time_column = next((col for col in df.columns if 'time' in col.lower()), None)
    if not time_column:
        raise ValueError("找不到时间列")

    # 转换时间格式
    df['datetime'] = pd.to_datetime(df[time_column])

    # 根据观测类型选择波段列并计算噪声
    if obs_type == 'carrier_phase':
        available_bands = []
        for req_col in ['L1C', 'L5Q']:
            matched_col = next((col for col in df.columns if req_col in col), None)
            if matched_col:
                df = df.rename(columns={matched_col: req_col})
                df[req_col] = pd.to_numeric(df[req_col], errors='coerce')
                available_bands.append(req_col)

        if not available_bands:
            raise ValueError("在数据中未找到任何可用的载波相位观测值列")

        # 如果指定了波段，检查它们是否在可用波段中
        if bands:
            selected_bands = [band for band in bands if band in available_bands]
            if not selected_bands:
                print(f"警告: 指定的波段 {bands} 在数据中不可用。可用波段: {available_bands}")
                print(f"将使用所有可用波段: {available_bands}")
                selected_bands = available_bands
        else:
            selected_bands = available_bands

        # 计算载波相位噪声
        df = calculate_carrier_phase_noise(df, selected_bands, filter_settings)

    else:  # 伪距
        available_bands = []
        for req_col in ['C1C', 'C5Q']:
            matched_col = next((col for col in df.columns if req_col in col), None)
            if matched_col:
                df = df.rename(columns={matched_col: req_col})
                df[req_col] = pd.to_numeric(df[req_col], errors='coerce')
                available_bands.append(req_col)

        if not available_bands:
            raise ValueError("在数据中未找到任何可用的伪距观测值列")

        # 如果指定了波段，检查它们是否在可用波段中
        if bands:
            selected_bands = [band for band in bands if band in available_bands]
            if not selected_bands:
                print(f"警告: 指定的波段 {bands} 在数据中不可用。可用波段: {available_bands}")
                print(f"将使用所有可用波段: {available_bands}")
                selected_bands = available_bands
        else:
            selected_bands = available_bands

        # 计算伪距噪声
        df = calculate_pseudorange_noise(df, selected_bands, filter_settings)

    # 绘制图形
    plot_noise(df, selected_bands, obs_type, filter_settings)


if __name__ == "__main__":
    # 方法1: 不带参数运行时，使用默认设置分析载波相位噪声
    # main()

    # 方法2: 通过命令行参数指定文件、波段和噪声类型
    # 例如: python noise_analysis.py --file "C:/data/GNSS_data.csv" --bands L1C --type carrier_phase
    # 例如: python noise_analysis.py --file "C:/data/GNSS_data.csv" --bands C1C C5Q --type pseudorange

    # 方法3: 直接在代码中指定参数
    # 载波相位噪声分析示例
    # main(
    #    input_file="C:/data/GNSS_data.csv",
    #    specified_bands=["L1C", "L5Q"],
    #    obs_type='carrier_phase',
    #    filter_settings={
    #        'cycle_slip_threshold': 0.3,
    #        'std_outlier_factor': 3.0,
    #    }
    # )

    # 伪距噪声分析示例
    main(
       input_file="C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\G08.csv",
       specified_bands=["C1C", "C5Q"],
       obs_type='pseudorange',
       filter_settings={
           'pr_jump_threshold': 5.0,
           'noise_meter_threshold': 100.0,
       }
    )


    # main(C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\G08.csv)