import os
import time
import concurrent.futures
import multiprocessing

import numpy as np

from answer import optimized

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 请参赛选手不要修改此文件
def get_score(phase_angle, amplitude, variables):
    """
    打分函数,请参赛选手不要修改此函数和其中调用的任何函数
    Args:
        phase_angle 相位角，32个元素的列表float，取值为0到2\pi的弧度制
        amplitude  阵子振幅，32个元素的列表float
        variables  参数变量列表，变量列表中包含3个元素
            第一个元素是 theta_0 信号方向，float数
            第二个元素是 相位量化比特数，取值为 1, 2, 3, 4
            第三个元素是 控制振幅是否优化，取值为True 或 False

    Returns：
        单个角度对应优化参数的分数
    """

    n_angle = 500
    N = 32
    theta_0 = variables[0]
    n_bit_phase = variables[1]
    opt_amp_or_not = variables[2]

    # 确保相位和振幅是按照赛题要求的取值
    if opt_amp_or_not is True:
        amplitude = amplitude / np.max(amplitude)
    else:
        amplitude = np.ones(N)
    phase_angle = np.angle(np.exp(1.0j * phase_angle)) + np.pi
    phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)

    efield = get_efield(n_angle, N, theta_0)
    theta_array = np.linspace(0, 180, 180 * n_angle + 1)
    amp_phase = []
    for i in range(N):
        amp_phase.append(amplitude[i] * np.exp(1.0j * phase_angle[i]))
    F = np.einsum('i, ij -> j', np.array(amp_phase), efield)
    FF = np.real(F.conj() * F)
    db_array = 10 * np.log10(FF / np.max(FF))

    x = theta_array - theta_0
    value_list = []
    for i in range(theta_array.shape[0]):
        if abs(x[i]) >= 30:
            value_list.append(db_array[i] + 15)
    a = max(np.max(value_list), 0)

    target = np.max(db_array)
    for i in range(theta_array.shape[0]):
        if db_array[i] == target:
            max_index = i
            break

    theta_up = 180
    theta_down = 0
    theta_min_up = 180
    theta_min_down = 0
    if abs(theta_array[max_index] - theta_0) > 1:
        print(f'Incorrect beamforming direction: {theta_array[max_index]}, with target: {theta_0}')
        y = 0
        print(f'final score: {y}')

    else:
        for i in range(1, 10000):
            if db_array[i + max_index] <= -30:
                theta_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if db_array[-i + max_index] <= -30:
                theta_down = theta_array[-i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (
                    db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[-i + max_index] < db_array[-i - 1 + max_index]) and (
                    db_array[-i + max_index] < db_array[-i + 1 + max_index]):
                theta_min_down = theta_array[-i + max_index]
                break

        if theta_up == 180 or theta_down == 0:
            print(f'Failed to identify expected mainlobe.')
            y = 0
            print(f'final score: {y}')

        elif theta_min_up < theta_up or theta_min_down > theta_down:
            print(f'The intensity of mainlobe did not decrease to -30 dB')
            y = 0
            print(f'final score: {y}')

        else:
            W = theta_up - theta_down
            b = max(W - 6, 0)

            value_list_2 = []
            for i in range(theta_array.shape[0]):
                if abs(x[i]) <= 30 and (x[i] >= theta_min_up - theta_0 or x[i] <= theta_min_down - theta_0):
                    value_list_2.append(db_array[i] + 30)
            c = np.max(value_list_2)

            # 负分直接归为0分
            y = max(1000 - 100 * a - 80 * b - 20 * c, 0)
            print(f'W = {W}, a = {a}, b = {b}, c = {c}, final score: y = {y}')
    return y


def cosd_f(x):
    return np.cos(x * np.pi / 180)


def get_efield(n_angle, N, theta_0):
    theta = np.linspace(0, 180, 180 * n_angle + 1)
    x = 12 * ((theta - 90) / 90) ** 2
    E_dB = -1.0 * np.where(x < 30, x, 30)
    E_theta = 10 ** (E_dB / 10)
    EF = E_theta ** 0.5

    phase_x = 1j * np.pi * cosd_f(theta)
    AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])

    efield = EF[None, ...] * AF

    return efield


def get_single_score(variable):
    phase_angle, amp = optimized(variable)
    return get_score(phase_angle, amp, variable)


def judgment(timeout=120, variables=None):
    scores = []
    count_timeout = 0
    for variable in variables:
        single_score = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(get_single_score, variable)
            try:
                single_score = future.result(timeout=timeout)
            except Exception:
                count_timeout += 1
                for process in multiprocessing.active_children():
                    process.terminate()
                    process.join()
        scores.append(single_score)
    print(scores)
    return np.mean(scores), count_timeout


if __name__ == '__main__':
    single_param_timeout = 90  # 单个参数超时时间
    variable_list = [[112, 2, False], [80, 2, True]]  # 参数列表
    start = time.time()
    score, timeout_param = judgment(single_param_timeout, variable_list)
    end = time.time()
    print("use time：", str(end - start))
    if timeout_param:
        print(f"failed:{timeout_param}/{len(variable_list)} timeout")
    print("score:", "%.4f" % score)
