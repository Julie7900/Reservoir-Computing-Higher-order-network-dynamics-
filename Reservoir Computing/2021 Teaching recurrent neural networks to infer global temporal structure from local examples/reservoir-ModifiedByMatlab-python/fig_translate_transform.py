
import scipy.sparse as sp
import numpy as np
from numpy.linalg import eigvals
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from matplotlib.collections import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib as mpl

#%%
########################## ReservoirTanh类 ###################################
class ReservoirTanh:
    def __init__(self,A, B, C, rs, xs, cs, delT, gam):
        self.A = A
        # 矩阵参数
        self.A = A
        self.B = B
        self.C = C
        # # 固定点与状态
        self.rs = np.asarray(rs)
        self.xs = np.asarray(xs)
        self.cs = np.asarray(cs)

        # 偏置项 d = atanh(rs) - A*rs - B*xs - C*cs
        self.d = np.arctanh(rs) - A @ rs - B @ xs - C @ cs
        self.d = self.d.reshape(-1, 1)
        # 时间相关参数
        self.delT = delT
        self.gam = gam

        self.r = np.zeros((A.shape[0], 1))      # 当前储层状态 r 初始化为零向量
        self.R = None                   # 反馈矩阵初始化（后续在 predict 阶段设置）


    # 储层训练阶段（输入 x 和 控制 c）
    def train(self, x, c):
        nx = x.shape[1]
        D = np.zeros((self.A.shape[0], nx))
        D[:, 0] = self.r.flatten()
        # D[:, 0] = self.r
        print("." * 100)
        for i in range(1, nx):
            if i % (nx // 100) == 0:
                print("=", end="", flush=True)
            self.propagate(x[:, i - 1, :], c[:, i - 1, :])
            D[:, i] = self.r.flatten()
            # D[:, i] = self.r
        print("\n")
        return D

    # 储层预测阶段（仅输入控制信号）
    def predict_x(self, c, W):
        """
        储层反馈预测阶段（只使用控制信号）
        输入：
            c : ndarray (K×T×4) 控制信号
            W : ndarray (M×N) 训练得到的读出矩阵
        输出：
            D : ndarray (N×T) 储层状态矩阵
        """
        nc = c.shape[1]
        # 设置反馈矩阵 R = A + B*W
        self.R = self.A + self.B @ W
        D = np.zeros((self.R.shape[0], nc))
        D[:, 0] = self.r.flatten()
        # D[:, 0] = self.r

        print("." * 100)
        for i in range(1, nc):
            if i % (nc // 100) == 0:
                print("=", end="", flush=True)
            self.propagate_x(c[:, i - 1, :])
            D[:, i] = self.r.flatten()
            # D[:, i] = self.r
        print("\n")
        return D

    # Runge-Kutta 四阶积分（训练阶段）
    def propagate(self, x, c):
        k1 = self.delT * self.del_r(self.r, x[:, [0]], c[:, [0]])
        k2 = self.delT * self.del_r(self.r + k1 / 2, x[:, [1]], c[:, [1]])
        k3 = self.delT * self.del_r(self.r + k2 / 2, x[:, [2]], c[:, [2]])
        k4 = self.delT * self.del_r(self.r + k3, x[:, [3]], c[:, [3]])
        self.r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Runge-Kutta 四阶积分（预测阶段）
    def propagate_x(self, c):
        """Runge-Kutta 四阶积分（预测阶段）"""
        self.r = self.r.reshape(-1, 1)
        c = c.reshape(c.shape[0], 4)  # 确保第二维为4

        k1 = self.delT * self.del_r_x(self.r, c[:, [0]])
        k2 = self.delT * self.del_r_x(self.r + k1 / 2, c[:, [1]])
        k3 = self.delT * self.del_r_x(self.r + k2 / 2, c[:, [2]])
        k4 = self.delT * self.del_r_x(self.r + k3, c[:, [3]])

        # 防止 NumPy 扩展错误
        self.r = (self.r + (k1 + 2 * k2 + 2 * k3 + k4) / 6).reshape(-1, 1)


    # ODE 定义（训练阶段）
    def del_r(self, r, x, c):
        """训练阶段的状态方程"""
        r = r.reshape(-1, 1)
        x = x.reshape(-1, 1)
        c = c.reshape(-1, 1)
        val = self.A @ r + self.B @ x + self.C @ c + self.d
        return self.gam * (-r + np.tanh(val))

    # ODE 定义（预测阶段）
    def del_r_x(self, r, c):
        """预测阶段的状态方程"""
        r = r.reshape(-1, 1)
        c = c.reshape(-1, 1)
        val = self.R @ r + self.C @ c + self.d
        return self.gam * (-r + np.tanh(val))


#%%
########################## Lorenz类 ###################################
class Lorenz:
    """
    洛伦兹系统类 (Lorenz system)
    用于生成驱动储层的混沌信号。
    对应 MATLAB 中的 Lorenz < handle 类。
    """

    def __init__(self, x0, delT, parms):
        """
        初始化洛伦兹系统参数
        参数说明：
        x0 : ndarray (3×1)  初始状态
        delT : float         仿真时间步长
        parms : list/tuple   系统参数 [sigma, rho, beta]
        """
        self.x0 = np.array(x0).reshape(3, 1)
        self.x = np.copy(self.x0)
        self.delT = delT
        self.parms = parms  # [σ, ρ, β]

    def propagate(self, n):
        """
        使用四阶 Runge-Kutta 方法积分 n 步
        输出：
            X : ndarray (3×n×4)
                X[:, :, 1] 保存主状态轨迹
                X[:, i-1, 2:4] 保存中间步，用于 RK4 与储层耦合
        """
        sigma, rho, beta = self.parms
        nInd = 0
        X = np.zeros((3, n, 4))
        X[:, 0, 0] = self.x.flatten()

        print("." * 100)

        # 定义洛伦兹系统的微分方程 dx/dt
        def dxdt(x):
            """Lorenz 系统微分方程"""
            x1, x2, x3 = x.flatten()
            return np.array([
                sigma * (x2 - x1),
                x1 * (rho - x3) - x2,
                x1 * x2 - beta * x3
            ]).reshape(3, 1)

        for i in range(1, n):
            if i > nInd * n:
                print("=", end="", flush=True)
                nInd += 0.01

            # 四阶 Runge–Kutta 积分
            k1 = self.delT * dxdt(self.x)
            k2 = self.delT * dxdt(self.x + k1 / 2)
            k3 = self.delT * dxdt(self.x + k2 / 2)
            k4 = self.delT * dxdt(self.x + k3)

            # 更新状态
            self.x = self.x + (k1 + 2*k2 + 2*k3 + k4) / 6

            # 保存当前状态和中间结果
            X[:, i, 0] = self.x.flatten()
            X[:, i - 1, 1:4] = np.column_stack([
                (self.x + k1 / 2).flatten(),
                (self.x + k2 / 2).flatten(),
                (self.x + k3).flatten()
            ])

        print("\n")
        return X

def downsample_curvature(X, a, v=None):
    """
    downsample_curvature: 下采样曲率特征轨迹
    参数:
        X : np.ndarray
            输入轨迹，形状为 (3, N)
        a : float
            曲率阈值，越大表示压缩越强
        v : list 或 tuple，可选
            若提供，则为观察角度 [azimuth, elevation] (单位：度)
            用于将轨迹投影到该视角平面（论文中用于绘图）

    返回:
        XC : np.ndarray
            压缩后的轨迹 (3, M)
        dInd : np.ndarray
            被保留的索引
    """
    if v is not None:
        XS = X.copy()
        # 构建视角向量 vx
        v = np.radians(v)  # 转换为弧度
        vx = np.array([
            np.sin(v[0]) * np.cos(v[1]),
            -np.cos(v[0]) * np.cos(v[1]),
            np.sin(v[1])
        ])

        # 计算 vx 的正交基
        vyz = np.linalg.svd(vx.reshape(1, -1))[2][1:3, :]
        # 投影到 2D 平面
        X = vyz @ X

    VInd = [1, 1]
    n_start = X.shape[1]
    aInd = np.arange(n_start)
    dInd = []

    # 主循环：逐步剔除曲率过低的点
    while len(VInd) > 1:
        V = np.diff(X, axis=1)
        # 向量夹角余弦（近似局部曲率）
        VMag = np.sum(V[:, 1:] * V[:, :-1], axis=0) / np.sqrt(np.sum(V[:, :-1]**2, axis=0))
        VNorm = np.concatenate([[a + 1], np.sqrt(np.sum(V[:, 1:]**2, axis=0) - VMag**2)])

        # 找出曲率低于阈值 a 的点
        VIndL = np.where(VNorm < a)[0]
        VIndU = np.where(VNorm > a)[0]

        # 移除过度锐利点附近的索引
        VIndL = np.setdiff1d(VIndL, np.concatenate([VIndU+1, VIndU-1]))
        if VIndL.size == 0:
            break

        # 寻找连续点组
        VIndD = np.diff(np.concatenate([[2], VIndL]))
        VIndNC = VIndL[VIndD != 1]       # 非连续索引
        VIndC = VIndL[VIndD == 1]        # 连续索引

        # 候选要移除的点
        VIndR = np.unique(np.concatenate([VIndNC, VIndC[::2]]))
        # 候选要保留的点
        VIndK = np.setdiff1d(np.arange(X.shape[1]), VIndR)

        # 计算保留点的曲率
        XP = X[:, VIndK]
        Vp = np.diff(XP, axis=1)
        VMag = np.sum(Vp[:, 1:] * Vp[:, :-1], axis=0) / np.sqrt(np.sum(Vp[:, :-1]**2, axis=0))
        VNorm = np.concatenate([[a], np.sqrt(np.sum(Vp[:, 1:]**2, axis=0) - VMag**2)])
        VIndKO = VIndK[VNorm > a]

        # 移除仍不合格的点
        VIndR = np.setdiff1d(VIndR, np.concatenate([VIndKO+1, VIndKO-1]))
        if VIndR.size == 0:
            break

        # 累积保留索引
        dInd.extend(aInd[VIndR])
        dr = np.setdiff1d(np.arange(X.shape[1]), VIndR)
        aInd = aInd[dr]
        X = X[:, dr]

    # 最终索引（保留的点）
    dInd = np.setdiff1d(np.arange(n_start), np.array(dInd))

    if v is not None:
        X = XS[:, dInd]

    XC = X[:, :]
    print(f"compression ratio: {XC.shape[1] / n_start:.3f}")
    return XC, dInd





##########################  ###################################

#%%
# #参数
delT = 0.001                # 模拟步长（时间分辨率）
t_waste = 20                # 暂态时间
t_train = 200               # 有效训练时间
n_w = int(t_waste/delT)     # 暂态样本数
n_t = int(t_train/delT)     # 训练样本数（200/0.001=200000步）
n = n_w + n_t               # 每个变换序列的总样本数

ind_t = (np.arange(n_t) + n_w).reshape(1, -1)    #注意后续索引会，matlab是20001-220000，这里是20000-219999
# 跨4个平移示例的索引：对应c=0,1,2,3（训练用的4个输入）
# t_ind = np.concatenate([ind_t, ind_t + n, ind_t + 2*n, ind_t + 3*n]).reshape(1, -1)
t_ind = np.concatenate([ind_t, ind_t + n, ind_t + 2*n, ind_t + 3*n]).ravel()
print(np.shape(t_ind))
#%%
# 储备池与洛伦兹系统核心参数（对应论文方程1、2）
N = 450                             # 水库神经元数量（论文中N=450）
M = 3                               # 洛伦兹系统维度（x1,x2,x3，论文方程1）
gam = 100                           # 水库响应性（论文方程2中的γ，时间常数倒数）
sig = 0.008                         # 吸引子影响强度
c = 0.004                           # 控制参数（对应论文中c）
p = 0.1                             # 水库连接密度（10#连接率，保证稀疏性）
x0 = np.zeros((M,1))                # 洛伦兹系统平衡点
c0 = np.zeros((1,1))                # 控制参数平衡点

#%%
"""
# 初始化储层与Lorenz随机参数
# 储层连接矩阵 A (N, N)
A = (np.random.rand(N, N) - 0.5) * 2     # 生成 [-1, 1] 区间的随机数矩阵
A = A * (np.random.rand(N, N) <= p)      # 稀疏化（按密度 p）
# 归一化谱半径（<1，确保储层稳定）
eig_max = np.max(np.real(eigvals(A)))
A = A / eig_max * 0.95
A = sp.csr_matrix(A)                     # 转换为稀疏矩阵（节省内存，加速运算）

B = 2 * sig * (np.random.rand(N, M) - 0.5)  # [-sig, sig] 区间    # 输入矩阵 B(N, M)（连接输入 x 到储层，对应论文公式(2)）
C = 2 * c * (np.random.rand(N, 1) - 0.5)    # [-c, c] 区间        # 控制参数矩阵 C(N, 1)（连接控制信号 c 到储层，对应论文公式(5)）
# 储层初始状态 r0（随机偏移，确保初始状态多样性）
r0 = (np.random.rand(N, 1) * 0.2 + 0.8) * np.sign(np.random.rand(N, 1) - 0.5)
Lx0 = np.random.rand(3, 1) * 10             # 洛伦兹系统初始条件（随机生成）
"""

##这里的fig_translate_transform_params.mat包含了A,B,C等参数，原文的图所采用的参数和数值
mat_data = sio.loadmat(r"fig_translate_transform_params.mat")
A   = mat_data['A']             # 储层连接矩阵 (450x450, 稀疏矩阵)
B   = mat_data['B']             # 输入矩阵 (450x3)
C   = mat_data['C']             # 控制矩阵 (450x1)
r0  = mat_data['r0']            # 储层初始状态 (450x1)
Lx0 = mat_data['Lx0']           # Lorenz 初始状态 (3x1)
c0  = mat_data['c0']            # 控制信号平衡点 (1x1)
delT = float(mat_data['delT'])  # 时间步长
gam  = float(mat_data['gam'])   # 响应参数

# 确保 A 是稀疏矩阵
if not sp.issparse(A):
    A = sp.csr_matrix(A)

# 2. 创建 Reservoir + Lorenz 对象
R2 = ReservoirTanh(A, B, C, r0, np.zeros((3,1)), c0, delT, gam)
L0 = Lorenz(Lx0, delT, [10, 28, 8/3])

#%%
# ################################ 创建系统对象和平移预测 ##############################################
"""
R2 = ReservoirTanh(A, B, C, r0, x0, c0, delT, gam)              # 储层系统
L0 = Lorenz(Lx0, delT, [10, 28, 8/3])                   # Lorenz 系统
"""



# Lorenz时间序列
print("Simulating Attractor...")            #模拟吸引子
X0 = L0.propagate(n)                        # X0.shape = (3, n, 4)
# 定义平移方向矩阵 P
a = np.array([[1.0], [0.0], [0.0]])         # 沿 x1 方向平移
a =a.reshape(3,1,1)                         # 3×1×1  ⚠⚠⚠⚠注意这里的广播机制⚠⚠⚠

## 生成 4 条平移后的轨迹
X1Ts = X0 + a                   # c=1 平移
X2Ts = X0 + 2 * a               # c=2 平移
X3Ts = X0 + 3 * a               # c=3 平移
print(X0.shape)                 # 验证矩阵维度
XinTs = np.concatenate([X0, X1Ts, X2Ts, X3Ts], axis=1)          # (3, 4n,4) # 拼接所有平移轨迹
print(f"Lorenz 序列生成完成: 原始轨迹形状 {X0.shape}，拼接后 {XinTs.shape}")

print(np.shape(XinTs[:, t_ind, 0]))

## 生成 4 条线性变换后的轨迹
I = np.eye(3)                   # 单位矩阵
T = np.zeros((3, 3))
T[0, 0] = -0.012                # 变换矩阵P，仅x1方向压缩
X1Tf = np.zeros_like(X0)
X2Tf = np.zeros_like(X0)
X3Tf = np.zeros_like(X0)
for i in range(4):                                              # 注意：X0[:, :, i] 是 (3, n)，需要矩阵乘法 (3,3) @ (3,n)
    X1Tf[:, :, i] = (I + T) @ X0[:, :, i]                       # c=1：x1压缩0.988倍
    X2Tf[:, :, i] = (I + 2*T) @ X0[:, :, i]                     # c=2：x1压缩0.976倍
    X3Tf[:, :, i] = (I + 3*T) @ X0[:, :, i]                     # c=3：x1压缩0.964倍
XinTf = np.concatenate([X0, X1Tf, X2Tf, X3Tf], axis=1)    # 拼接4个变换序列（训练输入：c=0,1,2,3），按第三维拼接
print(f"压缩拼接后 {XinTf.shape}")


#%%
# 四、储层训练（对应论文 2.2 节权重学习与反馈闭环）
Cin = np.ones((1,n,4))                                            # 控制信号Cin：每个序列对应一个c值（0,1,2,3），与输入序列匹配
Cin = np.concatenate([0*Cin, 1*Cin, 2*Cin, 3*Cin],axis=1)   # Cin维度：[1, n*4, 4]，对应4个序列的c值
print(f"Cin的形状 {Cin.shape}")

############# 1、平移任务 ###########
print("Simulating Reservoir...（1）")
RT = R2.train(XinTs, Cin)                           # 前向传播：生成储层状态矩阵 r(t)
RT = RT[:, t_ind]                                   # 跳过暂态部分，仅保留有效训练样本
print("Training W...（1）")
# 最小二乘解：WTs = argmin_W ||W*RT - XinTs||²
# 对应 MATLAB: WTs = lsqminnorm(RT', XinTs(:,t_ind,1)')'
# print(np.shape(XinTs[:, t_ind, 0]))                               #验证矩阵维数（3*800000）
WTs, *_ = np.linalg.lstsq(RT.T, XinTs[:, t_ind, 0].T, rcond=None)
WTs = WTs.T                                                         # 转置以保持维度一致 (输出维 × reservoir size)
XTs = WTs @ RT                                                      # 训练输出：预测结果 XTs = WTs * RT
train_error = np.linalg.norm(XTs - XinTs[:, t_ind, 0])              # 计算训练误差（2-范数）
print(f"Training error: {train_error}")
rsTs = RT[:, n_t]                                   # 训练结束时的水库最终状态
del RT                                              # 清理（Python中通常不需要）

############# 2、线性变换 ############
print("Simulating Reservoir...（2）")
RT = R2.train(XinTf, Cin)                           # 用变换序列XinTf训练 reservoir
RT = RT[:, t_ind]                                   # 按索引取训练样本（确保 t_ind 是 1D）
print("Training W...（2）")
# ===== 最小二乘求解 =====
# 对应 MATLAB 的 lsqminnorm(RT', XinTf(:,t_ind,1)')'
# 即求解 WTf，使得 WTf @ RT ≈ XinTf[:, t_ind, 0]
WTf, *_ = np.linalg.lstsq(RT.T, XinTf[:, t_ind, 0].T, rcond=None)   # 最小二乘解
WTf = WTf.T                                                         # 转置，与 MATLAB 保持一致
XTf = WTf @ RT                                                      # 模拟输出
train_error = np.linalg.norm(XTf - XinTf[:, t_ind, 0])
print(f"Training error: {train_error}")
rsTf = RT[:, n_t]                                                   # 保存 reservoir 最终状态对应 MATLAB: RT(:, n_t)

#%%
# 五、预测与外推（对应论文 3.1 节 extrapolation 结果）
# 1. 初始化参数与 reservoir 状态
print("Generating Reservoir Prediction: Translate")     # 平移
R2.r = rsTs                             # 重置 reservoir 内部状态为训练后的最终状态（rsTs）
nR = 40000                              # 停留时间步
nT = 40000                              # 移动时间步
nVS, nVC = 40, 10                       # 示例（MATLAB 中是变量）
# 2. 构造控制信号 cInds1Ts
cInds1Ts = np.concatenate([
    np.linspace(0, -nVS, nT),      # 从 0 开始匀速“移动”到负的极限 -nVS（第一个 ramp）。
    -nVS * np.ones(nR + nT),            # 在 -nVS 位置停留一段时间（停留时间是 nR + nT 步）
    np.linspace(-nVS, -nVC, nT),        # 从更远的负值向接近中间的负值移动（第二个 ramp）
    -nVC * np.ones(nR),                 # 在 -nVC 位置停留 nR 步。
    np.linspace(-nVC, nVC, 2 * nT),     # 穿越从负中值到正中值（中间可能代表快速变化区间）。
    nVC * np.ones(nR),                  # 在 +nVC 处停留 nR 步。
    np.linspace(nVC, nVS, nT),          # 从 +nVC 线性变化到更远的正值 +nVS，共 nT 个点（向外延伸的 ramp）。
    nVS * np.ones(nR)                   # 在 +nVS 位置停留 nR 步。
])

# 3. 控制信号微分（cDiff1aTs）
cDiff1aTs = np.diff(cInds1Ts, prepend=cInds1Ts[0])
cDiff1aTs[-1] = 0  # 保持长度一致
# 4. 构造 reservoir 输入格式 (1, N, 4)
# 对应 MATLAB reshape([c; c+diff/2; c+diff/2; c]', [1, len, 4])
cInds1aTs = np.stack([
    cInds1Ts,
    cInds1Ts + cDiff1aTs / 2,
    cInds1Ts + cDiff1aTs / 2,
    cInds1Ts
], axis=-1).reshape(1, -1, 4)
# 5. 预测阶段
RCont = R2.predict_x(cInds1aTs, WTs)
# 6. 取稳定后的部分（丢弃前 2*nT 步）
RCont = RCont[:, 2 * nT:]          # reservoir 状态
cDiff1Ts = cDiff1aTs[2 * nT:]      # 对应差分
cInds1aTs_cut = cInds1aTs[:, 2 * nT:, 0]  # 对应控制信号
# 7. 输出预测结果（洛伦兹吸引子平移轨迹）
XCTs = WTs @ RCont
print(f"Prediction completed. XCTs shape: {XCTs.shape}")



#####
# 1. 初始化参数与 reservoir 状态
print("Generating Reservoir Prediction: Transform")
R2.r = rsTf.copy()    # 初始化 reservoir 状态（训练结束时的状态）
nR = 40000            # 固定边界保持步数
nT = 300000           # 扫描的主时间长度
nVS = 40
# 2.构造参数 cInds1Tf：拼接 [-nVS, nVS] 的扫描路径
cInds1Tf = np.concatenate([
    np.linspace(0, -nVS, nT),
    -nVS * np.ones(nR),
    np.linspace(-nVS, nVS, nT),
    nVS * np.ones(nR)
])
# 3.一阶差分（相邻差）
cDiff1Tf = np.concatenate([np.diff(cInds1Tf), [0]])
# 4.构造四阶段输入（RK4 对应 c 的 4 个阶段）
cInds1aTf = np.stack([
    cInds1Tf,
    cInds1Tf + cDiff1Tf / 2,
    cInds1Tf + cDiff1Tf / 2,
    cInds1Tf
], axis=-1).reshape(1, len(cInds1Tf), 4)
# 5.Reservoir 预测阶段
RCont = R2.predict_x(cInds1aTf, WTf)
# 6.丢弃前段暂态，保留后半段轨迹
RCont = RCont[:, nT:]             # 对应 MATLAB 的 (1*nT+1:end)
cDiff1Tf = cDiff1Tf[nT:]
cInds1Tf = cInds1Tf[nT:]
# 7.输出层映射：Lorenz 吸引子变换轨迹
XCTf = WTf @ RCont                # shape: (3, time_steps)
print("Transform trajectory generated, shape:", XCTf.shape)


#%% ================= 可视化部分 =================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -----------------------------
colors = [
    (0.05, 0.05, 0.5),   # 深蓝 navy
    (0.2, 0.3, 0.9),     # 蓝 blue
    (0.0, 0.8, 0.7),     # 青 cyan-green
    (0.0, 0.9, 0.3)      # 亮绿 light green
]
nature_cmap = LinearSegmentedColormap.from_list("nature_bluegreen", colors, N=256)
norm = Normalize(vmin=-40, vmax=40)

def plot_gradient_curve(X_pred, c_vals, title, filename):
    """
    绘制二维渐变颜色曲线
    X_pred: shape (2, N) 或 (3, N)
    c_vals: 对应每个点的 c 值
    """
    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    # 将曲线拆分为线段
    points = np.array([X_pred[0], X_pred[1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 连续渐变曲线
    lc = LineCollection(segments, cmap=nature_cmap, norm=norm, linewidths=1.2, alpha=0.45)
    lc.set_array(c_vals)
    ax.add_collection(lc)

    ax.set_xlim(X_pred[0].min(), X_pred[0].max())
    ax.set_ylim(X_pred[1].min(), X_pred[1].max())

    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

    # 颜色条，明确指定 ax=ax
    sm = plt.cm.ScalarMappable(cmap=nature_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.02)
    cbar.set_label('c', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# -----------------------------
# 平移表示
plot_gradient_curve(XCTs, cInds1Ts.flatten(), '(b) Translate representation', 'figure2-b_2D.svg')

# 压缩表示
plot_gradient_curve(XCTf, cInds1Tf.flatten(), '(d) Compressed representation', 'figure2-d_2D.svg')
