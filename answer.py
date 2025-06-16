 ## 关于相位编码，可以参考文献：https://arxiv.org/pdf/2409.19938 ##

import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

def optimized(variables):
    """
    相位振幅优化函数，请勿修改或删除函数名称，请勿修改函数的返回值，否则判分失败
    Args:
        variables 一个包含3个元素的列表；
        列表中的第1个元素代表波束成形方向theta_0，45 <= theta_0 <= 135
        列表中的第2个元素代表相位角度取多少比特离散值，可能取值为1,2,3,4
        列表中的第3个元素是控制振幅是否优化，取值为True 或 False

    Returns：
        phase_angle 相位角，32个元素的列表float
        amp 阵子振幅，32个元素的列表float

    本样例代码中采用torch微分库实现了偏导计算。
    """

    # 在后续优化过程中使用的参数
    param = {
        'theta_0': variables[0],  # 波束成形方向，以90度为例
        'N': 32,  # 天线阵子总数
        'n_angle': 10,  # 1度中被细分的次数
        'encode_qubit': variables[1],  # 进行相位编码的比特个数，样例代码中固定为2，实际对应变量 variables[1]

        # bSB参数界面
        'xi': 0.1,  # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.3,  # 演化步长
        'n_iter': 2000,  # 迭代步数

        # 相位损失函数界面 
        'weight': 0.01,  # 调节损失函数中分子和分母的相对大小
        'range_list': [[-30, -6], [6, 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
        'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

        # 连续振幅优化
        'opt_amp_or_not': variables[2], # 根据输入控制是否优化振幅
        'lr': 0.001, # 学习率
    }

    A = BF(param)
    A.solve()
    A.plot()

    phase_angle = A.phase_angle
    amp = A.amp

    return phase_angle, amp


"""
下面为样例代码
"""

# 计算角度制下的三角余弦函数
def cosd_f(x):
    return np.cos(x * np.pi / 180)

# 计算角度制下的三角正弦函数
def sind_f(x):
    return np.sin(x * np.pi / 180)


# 基于bSB或dSB算法进行波束赋形
class BF():
    '''
        此为基于量子启发算法进行优化的核心部分。
        样例代码进给出了一种优化思路，显然有很多方法可以进一步改善波束赋形的结果，包括但不限于：
        1、合理设置SB算法中的超参数，比如演化步长dt、迭代步数n_iter、损失函数控制系数xi等；
        2、合理选取其他形式的损失函数和损失函数中的超参数，样例代码中给出了一种参考目标函数的形式；
        3、将量子启发算法与其他优化策略结合；
        4、采用模拟分叉（SB）方法之外的其他量子启发算法，比如LQA，模拟伊辛机等。
        5、选手可以重点改进样例代码中相位和振幅的优化流程函数、纯相位优化函数、纯振幅优化函数、相位比特编码函数这些函数。

        其中主要变量分别为如下形式：
        x, y, x_bit: 2维向量, size: param['N'] * param['encode_qubit']
        amp, phase_angle: 1维向量, size: param['N']
        efield: 2维向量, size: param['N'] * (180 * param['n_angle'] + 1)

    '''
    # 初始化变量
    # ============================================================================================================================ #
    def __init__(self, param):
        self.param = copy.deepcopy(param)
        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        self.amp = np.ones(self.param['N'])
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

    # 相位和振幅的优化求解流程
    # ============================================================================================================================ #
    def solve(self):
        # 优化相位角
        self.x_final = self.opt_phase()
        self.phase = self.encode(self.x_final)
        self.phase_angle = np.angle(self.phase)

        # 优化振幅
        if self.param['opt_amp_or_not'] is True:
            self.amp, _ = self.opt_amp(self.amp, self.phase_angle)
            self.amp = np.array(self.amp.clone().detach().numpy())

        # 后处理：归一化振幅，并且将振幅按照量化比特数的要求进行离散化；将负数振幅转化为正数的形式，同时给相位角加上pi
        cond = self.amp < 0
        self.phase_angle = np.angle(np.exp(1.0j * np.where(cond, self.phase_angle + np.pi, self.phase_angle)))
        self.phase_angle = self.phase_angle + np.pi # 确保最后相位角变化为0到2\pi
        self.amp = np.abs(self.amp) / np.max(np.abs(self.amp)) # 将振幅归一化
        print(f'phase angle: {self.phase_angle}')
        print(f'amp: {self.amp}')
        
    # 基于模拟分叉方法的纯相位优化函数
    # ============================================================================================================================ #
    def opt_phase(self):
        if self.param['encode_qubit'] == 2:
            from mindquantum.algorithm.qaia import BSB
            '''
                对于QUBO问题，也可直接使用mindquantum中QAIA模块中的相关功能；
                我们采用2比特编码相位编码并忽略振幅，展示通过QAIA中的BSB模块实现相位优化的思路，为了取得更好的优化效果需要进一步改进代码；
                在这里，我们将损失函数主瓣信号强度减去加权的旁瓣信号强度写成哈密顿矩阵J和表示相位的向量x的内积的形式，进而匹配QAIA算法中的相应输入形式（比如这里用的BSB模块）。 
            '''
            c1 = 0.5 + 0.5j
            c2 = 0.5 - 0.5j
            # 针对32个相位，对每个相位采用2比特编码，进而用64个实数变量描述损失函数，并且构建对应的64*64的J矩阵
            factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1,
                                      self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
            J_enhance = torch.einsum('i, j -> ij ', factor_array.conj(), factor_array)
            J_suppress = 0.0
            for i in range(len(self.param['range_list_weight'])):
                num = 0
                a_0 = 0.0
                for j in range(round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])),
                               round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1])), 1):
                    num += 1
                    factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                    a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
                J_suppress += self.param['range_list_weight'][i] * a_0 / num
            J = torch.real((self.param['weight'] * J_enhance - (
                        1 - self.param['weight']) * J_suppress)).numpy()  # 由矩阵的构造可知J[i, j] = J*[j, i]，因此取实部不影响计算结果

            # 根据损失函数的形式完成J矩阵构建后调用mindquantum中QAIA的BSB模块进行优化
            solver = BSB(np.array(J, dtype="float64"), batch_size=1)
            solver.update()
            x_bit = np.sign(solver.x.reshape(2 * self.param['N'], 1))

            return x_bit.reshape(self.param['N'], self.param['encode_qubit'],
                                 order='F')  # 将x_bit的形式统一为 self.param['N'] * self.param['encode_qubit'] 这一二维矩阵的形式

        else:
            # 优化相位的损失函数
            def cost_func(x_bit):
                '''
                Args:
                    x_bit: SB算法中的变量x
                Returns:
                    obj: 损失函数的数值
                '''
                phase = self.encode(x_bit)
                amp = torch.tensor(self.amp.copy())

                main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
                loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
                loss_1 = 0.0
                for i in range(len(self.param['range_list_weight'])):
                    one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                        self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                        self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                    loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                        torch.real(torch.conj(one_range) * one_range)).mean()
                obj = loss_1 / loss_2 # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
                return obj

            # 初始化
            x = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))
            y = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))

            for iter in tqdm(range(self.param['n_iter'] + 1)):
                x_torch = torch.tensor(x)
                x_torch.requires_grad = True

                # 计算梯度与更新参数
                x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach() #dSB
                loss = cost_func(x_sign)
                loss.backward()
                x_grad = (x_torch.grad).clone().detach().numpy()
                y += (-(0.5 - iter / self.param['n_iter']) * x - self.param['xi'] * x_grad / np.linalg.norm(x_grad)) * self.param['dt']
                x = x + y * self.param['dt']
                cond = np.abs(x) > 1
                x = np.where(cond, np.sign(x), x)
                y = np.where(cond, np.zeros_like(y), y)

            return np.sign(x)


    # 纯振幅优化函数
    # ============================================================================================================================ #
    def opt_amp(self, amp, phase_angle):
        '''
        Args: 
            amp: 优化前的振幅
            phase_angle: 固定的相位角
        Returns:
            amplitude: 优化过后得到的振幅
            loss: 优化过后损失函数的数值
        '''
        # 优化振幅的损失函数
        def cost_func_for_amp(amp, phase_angle):
            '''
            Args: 
                amp: 振幅
                phase_angle: 相位角
            Returns:
                obj: 损失函数的数值
            '''
            phase = torch.exp(1.0j * phase_angle)

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 1.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()

            obj = loss_1 / loss_2
            return obj

        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])

        for iter in tqdm(range(1000)):
            optimizer.zero_grad()
            loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()
        return amplitude, loss

    # 辅助函数
    # ============================================================================================================================ #

    # 相位比特编码函数，可以参考文献：https://arxiv.org/pdf/2409.19938 
    def encode(self, x_bit):
        '''
        Args: 
            x_bit: 编码前的比特串
        Returns:
            phase: 编码后的相位

        此编码函数仅针对2比特编码的情况
        '''
        if self.param['encode_qubit'] == 1:
            N = x_bit.shape[0]
            phase = x_bit[:, 0]

        if self.param['encode_qubit'] == 2:
            c0 = 0.5 + 0.5j
            c1 = 0.5 - 0.5j
            N = x_bit.shape[0]
            phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]

        if self.param['encode_qubit'] == 3:
            A = np.sqrt(4 + 2 * np.sqrt(2)) / 4
            B = np.sqrt(4 - 2 * np.sqrt(2)) / 4
            c0 = A * np.exp(1j * 3 * np.pi / 8)
            c1 = A * np.exp(-1j * np.pi / 8)
            c2 = B * np.exp(-1j * np.pi / 8)
            c3 = B * np.exp(-1j * 5 * np.pi / 8)
            N = x_bit.shape[0]
            phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1] + c2 * x_bit[:, 2] + c3 * x_bit[:, 0] * x_bit[:, 1] * x_bit[:, 2]

        if self.param['encode_qubit'] == 4:
            def calculate_phase_coefficients():
                # 固定s1=1，生成s2, s3, s4的8种组合
                bit_combinations = np.array([
                    [1, 1, 1],  # s2=1, s3=1, s4=1
                    [1, 1, -1],  # s2=1, s3=1, s4=-1
                    [1, -1, 1],  # s2=1, s3=-1, s4=1
                    [1, -1, -1],  # s2=1, s3=-1, s4=-1
                    [-1, 1, 1],  # s2=-1, s3=1, s4=1
                    [-1, 1, -1],  # s2=-1, s3=1, s4=-1
                    [-1, -1, 1],  # s2=-1, s3=-1, s4=1
                    [-1, -1, -1]  # s2=-1, s3=-1, s4=-1
                ])

                # 构建比特矩阵S_N
                S_N = np.zeros((8, 8))
                for i, (s2, s3, s4) in enumerate(bit_combinations):
                    # 一阶项: 1, s2, s3, s4
                    # 三阶项: s2*s3, s2*s4, s3*s4, s2*s3*s4
                    S_N[i] = [1, s2, s3, s4, s2 * s3, s2 * s4, s3 * s4, s2 * s3 * s4]

                # 计算相位向量P (8个相位)
                P = np.exp(1j * np.pi * np.arange(0, 16, 2) / 8)  # θ = 0, π/4, π/2, ..., 7π/4

                # 求解系数向量C: C^T = S_N^(-1) * P^T
                try:
                    S_N_inv = np.linalg.inv(S_N)
                    C = np.dot(S_N_inv, P.conj()).conj()  # 取共轭以匹配公式中的转置关系
                    return C
                except np.linalg.LinAlgError:
                    print("矩阵S_N不可逆，无法求解系数")
                    return None

            # 执行计算并获取c0-c7
            coefficients = calculate_phase_coefficients()

            if coefficients is not None:
                # 将结果存入c0到c7变量中
                c0, c1, c2, c3, c4, c5, c6, c7 = coefficients
                # 现在可以在后续代码中使用c0-c7
                # 例如：print(f"计算得到的系数c0为: {c0}")
                c0, c7 = c7, c0
                c1, c6 = c6, c1
                c2, c5 = c5, c2
                c3, c4 = c4, c3
            N = x_bit.shape[0]
            phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1] + c2 * x_bit[:, 2] + c3 * x_bit[:, 3] + c4 * x_bit[:, 0] * x_bit[:, 1] * x_bit[:, 2] + c5 * x_bit[:, 0] * x_bit[:, 1] * x_bit[:, 3] + c6 * x_bit[:, 0] * x_bit[:, 2] * x_bit[:, 3] + c7 * x_bit[:, 1] * x_bit[:, 2] * x_bit[:, 3]

        phase.reshape(N)
        return phase
    
    # 优化结果画图函数
    def plot(self):
        # 计算画图相关数据
        self.theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))

        # 画图
        plt.figure()
        plt.plot(self.theta, self.y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$' + ' (dB)')
        plt.title('Beamforming Outcome')
        plt.savefig(str(self.param['theta_0']) + '_beamforming.jpg')

    # 生成单元阵子的辐射电场强度（随角度变化的函数）
    def _generate_power_pattern(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta ** 0.5
        return EF

    # 生成阵因子A_n
    def _generate_array_factor(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        return AF

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value):
        index = round(angle_value * self.param['n_angle'])
        return index
