# 结构1：常规结构
# 三维数据预测的RC
#bayes参数调优

#2025.9.16 关于输入度k的优化问题，尚未解决...
#2025.9.17  输入度直接取整即可，另外，论文中，岭回归参数reg（论文中的alpha）不参与参数调优，
#           由于文中的岭回归参数未知，此处设为1e-5，
#           目前采用的平均rmse作为优化目标函数，可以预测接近10s（10个lyapunov时间），
#           但是采用文中的优化后参数，预测效果不佳。后面采用文中定义的误差指标epsilon1试试...
#20250926 换了求解器，改为了4阶龙格库塔,但预测效果一样的。

import numpy as np
import scipy.sparse as sp
from bayes_opt import BayesianOptimization
from scipy.linalg import solve
from scipy.integrate import odeint
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

# 设置固定随机种子，确保随机数可复现
np.random.seed(42)  # 种子值可任意指定，如42、100等

N = 100                       # 储备池节点数量
# rho_Wr = 0.8                  # 储备池权重矩阵的谱半径
# rho_in = 0.084                    # 输入强度
# dens = 0.02                   # 储备池权重矩阵的连接密度
# leak_rate = 0.6               # 泄露率
# bias = 1.6                    # 偏置
reg = 1e-7                  #正则化参数 岭回归参数
d = 3

lyapunov = 0.9
lyapunov_time = lyapunov
#生成数据集
sigma = 10
beta = 8/3
rho = 28
dt = 0.01
def lorenz(t,init_state):
    x,y,z = init_state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

t = np.arange(0,100,dt)
init_state = [1,1,1]
# data = odeint(lambda y,t:lorenz(t,y), init_state, t)
# data = data.T   #转化为3*T的
# print(np.size(data))

def rk4(f, t0, y0, h, n):
    """四阶龙格库塔方法"""
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0))) if isinstance(y0, (list, np.ndarray)) else np.zeros(n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)

        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y

t0 = 0
y0 = init_state
h = dt
n = len(t) - 1
t_rk4, data1 = rk4(lorenz,t0,y0,h,n)
print(np.shape(data1))
data = data1.T


##划分数据
warm = 100
# data = data[:,warm:]
train_ratio = 0.8
train_size = round(len(data.T) * train_ratio)
test_size = round(len(data.T) * (1-train_ratio))

train_size = round(len(data.T) * train_ratio)
test_time = 20
test_size = round(test_time/dt)


u_train = data[:, :train_size]  # 使用第一行(x)作为输入
u_test = data[:, train_size:train_size+test_size]

def rc_objective(gamma, sigma1, rho_in, k, rho_Wr, u_train, u_test, train_size, test_size, warm):

    '''
    gamma 储备池特征时间尺度（ inverse time scale）7–11        7.7 此处暂时替换为泄露率 leak_rate
    sigma1 节点与输入的连接概率0.1–1.0                             0.81
    rho_in 输入权重的尺度（方差开方）0.3–1.5                     0.37
    k 节点的递归入度（每个节点的输入节点数）1–5                    3（优化后固定）
    rho_Wr 内部连接矩阵的谱半径（递归强度）0.3–1.5              0.41
    reg 岭回归参数
    '''
    leak_rate = gamma *dt
    k = int(round(k))  # round确保四舍五入（若优化器返回7.9，转为8）
    # 限制k在[5,11]范围内（防止极端情况溢出）
    k = max(5, min(k, 11))  # 确保k不超出预设范围


    #初始化输入权重 W_in N*d
    win_mean = 0  # 均值
    win_varience = rho_in**2  # 方差
    win_std = np.sqrt(win_varience)  # 标准差
    W_in = np.random.normal(win_mean, win_std, size=(N, d))
    # 每个节点与输入信号（3 维，对应混沌系统的 3 个维度）连接的概率为sigma
    mask = (np.random.rand(N, 3) < sigma1).astype(float)
    W_in = W_in*mask

    # W_r
    # 定义非零连接节点node
    node = np.zeros((N, k),dtype = int)
    for i in range(N):
        # 无放回选择：从除自身外的N-1个节点中选k个（避免自连接，论文默认逻辑）
        available_nodes = np.delete(np.arange(N), i)  # 排除当前节点i
        node[i] = np.random.choice(available_nodes, size=k, replace=False)
    # 赋值
    W_r = np.zeros((N, N))
    for i in range(N):
        weights = np.random.normal(0, 1, size=k)  # 生成k个N(0,1)权重
        W_r[i, node[i]] = weights  # 将权重填入对应输入节点位置
    # 调整W_res的谱半径（使其最大特征值的绝对值=设定的spectral_radius）
    # 谱半径=最大特征值的绝对值，这一步是为了满足回声状态特性
    # W_r = rho_Wr * W_r / max(abs(eigvals(W_r)))
    # eigvals, _ = sp.linalg.eigs(W_r, k=1)  # k=1表示只计算最大的1个特征值
    eigvals, _ = sp.linalg.eigs(
        W_r,
        k=1,
        which='LM',  # 明确指定找模最大的特征值
        maxiter=10000,  # 增加迭代次数
        tol=1e-4  # 适当放宽精度要求
    )
    max_eig = np.abs(eigvals)
    if max_eig ==0:
        W_r = rho_Wr * W_r
    else:
        W_r =  rho_Wr *W_r / max_eig

    #%% 初始化储备池状态
    temp_state = np.zeros(N)       #这里temp_state不用改！仍然保持二维矩阵
    # 储备池状态，存储训练过程中所有时刻的储备池状态
    states = np.zeros((N,train_size))   #这里states不用改！仍然保持二维矩阵
    for t in range(1,train_size):
        input_signal = u_train[:,t-1]               # 当前训练输入（t时刻的x值）
        # 计算t时刻的储备池状态（Leaky ESN的状态更新公式）
        # 状态 = (1-泄露率)*上一时刻状态 + 泄露率*tanh(输入映射 + 储备池内部映射)
        temp_state = (1-leak_rate)*temp_state + \
                     leak_rate * np.tanh(W_in @ input_signal + W_r @ temp_state)   #形状为N*d   偏置+ bias * np.ones((N,1))
        states[:,t] = temp_state

    ##岭回归
    states = states[:, warm:]  # 稳定后的状态
    u_train = u_train[:,warm:]   # 对应的训练数据
    # W_out = u_train @ states.T @ np.linalg.pinv(states @ states.T + reg * np.eye(N))
    W_out = u_train @ states.T @ solve(states @ states.T + reg * np.eye(N), np.eye(states.shape[0]))
    #预测
    # #用储备池状态的最后作为预测的输入,转化为N维数组之后，再转化为N*1的二维矩阵的形式
    temp_state = states[:,-1]
    u_pred = np.zeros((d,test_size)) #用来存储预测的数据
    input_signal = u_train[:,-1]                   # 当前训练输入（t时刻的x值）
    for t in range(test_size):
        # input_signal = u_test[t,:].reshape(d,1)                       # 当前训练输入（t时刻的x值）
        # 计算t时刻的储备池状态（Leaky ESN的状态更新公式）
        # 状态 = (1-泄露率)*上一时刻状态 + 泄露率*tanh(输入映射 + 储备池内部映射)
        temp_state = (1-leak_rate)*temp_state + \
                     leak_rate*np.tanh(W_in @ (W_out @ temp_state) + W_r @ temp_state)
        u_pred[:,t] = (W_out @ temp_state).ravel()

    # 计算每个分量的RMSE（评估x/y/z的预测精度）
    rmse_x = np.sqrt(np.mean((u_test[:, 0] - u_pred[:, 0]) ** 2))
    rmse_y = np.sqrt(np.mean((u_test[:, 1] - u_pred[:, 1]) ** 2))
    rmse_z = np.sqrt(np.mean((u_test[:, 2] - u_pred[:, 2]) ** 2))
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    return -rmse  # 负号是因为贝叶斯优化默认是最大化

def wrapped_rc_objective(gamma, sigma1, rho_in, k, rho_Wr):
    return rc_objective(gamma, sigma1, rho_in, k, rho_Wr,
                        u_train, u_test, train_size, test_size, warm)

pbounds = {
    'gamma': (7, 11),  # 自然衰减速率
    'sigma1':(0.1,1.0),          # 输入权重矩阵W_in的连接概率
    'rho_in': (0.01, 0.5),  # 输入强度
    'k':(1,5),           # 输入度
    'rho_Wr': (0.1, 1.5),       # 储备池权重矩阵的谱半径
}

optimizer = BayesianOptimization(
    f = wrapped_rc_objective,
    pbounds = pbounds,
    random_state = 42,
    verbose = 2
)
optimizer.maximize(init_points=5,n_iter=20,)
best_params = optimizer.max['params']
print("最优参数:", best_params)

# =============== 用最优参数再跑一次并画图 =================
rho_Wr = best_params['rho_Wr']
rho_in = best_params['rho_in']
sigma1 = best_params['sigma1']
gamma = best_params['gamma']
k = best_params['k']

leak_rate = gamma * dt
# 关键：转换k为整数，防止浮点数（如5.0）
k = int(round(best_params['k']))
# 再次限制k范围（双重保险）
k = max(5, min(k, 11))

# 重新训练+预测
N = 100
np.random.seed(42)

# W_in
win_mean = 0  # 均值
win_varience = rho_in**2  # 方差
win_std = np.sqrt(win_varience)  # 标准差
W_in = np.random.normal(win_mean, win_std, size=(N, d))

# W_r
wr_mean = 0                     #均值
wr_varience = rho_Wr                 #方差
wr_std = np.sqrt(wr_varience)    #标准差
W_r = np.random.normal(wr_mean, wr_std, size=(N, N))
eigvals, _ = sp.linalg.eigs(W_r, k=1)
max_eig = np.abs(eigvals)
if max_eig == 0:
    W_r = rho_Wr * W_r
else:
    W_r = rho_Wr * W_r / max_eig

states = np.zeros((N, train_size))
temp_state = np.zeros(N)
for t in range(1,train_size):
    input_signal = u_train[:,t-1]
    temp_state = (1-leak_rate)*temp_state + \
                 leak_rate*np.tanh(W_in @ input_signal + W_r @ temp_state)
    states[:,t] = temp_state

states = states[:, warm:]
u_train = u_train[:, warm:]
W_out = u_train @ states.T @ solve(states @ states.T + reg * np.eye(N), np.eye(states.shape[0]))

temp_state = states[:,-1]
u_pred = np.zeros((d,test_size))
for t in range(test_size):
    temp_state = (1-leak_rate)*temp_state + \
                 leak_rate*np.tanh(W_in @ (W_out @ temp_state) + W_r @ temp_state)
    u_pred[:,t] = (W_out @ temp_state).ravel()

# 画图
t_test = np.arange(test_size) * dt
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
comp = ['x','y','z']
for i in range(3):
    axes[i].plot(t_test, u_test[i,:], 'b-', label=f' {comp[i]}')
    axes[i].plot(t_test, u_pred[i,:], 'r--', label=f' {comp[i]}')
    axes[i].set_ylabel(f'{comp[i]} ')
    axes[i].legend()
axes[0].set_title("Lorenz forecast")
axes[2].set_xlabel('time / s)')
plt.tight_layout()
plt.show()




'''
# 绘制x/y/z三个分量的真实值与预测值对比
t_test = np.arange(test_size) * dt  # 测试集时间轴
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

# x分量对比
axes[0].plot(t_test, u_test[0, :], 'b-', label='真实x', linewidth=1.5)
axes[0].plot(t_test, u_pred[0, :], 'r--', label='预测x', linewidth=1.5)
axes[0].set_ylabel('x 分量')
axes[0].legend()
axes[0].set_title(f'Lorenz三维预测结果（RMSE_x={rmse_x:.4f}）')

# y分量对比
axes[1].plot(t_test, u_test[1, :], 'b-', label='真实y', linewidth=1.5)
axes[1].plot(t_test, u_pred[1, :], 'r--', label='预测y', linewidth=1.5)
axes[1].set_ylabel('y 分量')
axes[1].legend()

# z分量对比
axes[2].plot(t_test, u_test[2, :], 'b-', label='真实z', linewidth=1.5)
axes[2].plot(t_test, u_pred[2, :], 'r--', label='预测z', linewidth=1.5)
axes[2].set_ylabel('z 分量')
axes[2].set_xlabel('时间 (s)')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"RMSE (x分量): {rmse_x:.4f}")
print(f"RMSE (y分量): {rmse_y:.4f}")
print(f"RMSE (z分量): {rmse_z:.4f}")

plt.figure()
plt.plot(u_test[:,0],u_test[:,2],'k-')
plt.plot(u_pred[:,0],u_test[:,2],'r--')
plt.show()
'''