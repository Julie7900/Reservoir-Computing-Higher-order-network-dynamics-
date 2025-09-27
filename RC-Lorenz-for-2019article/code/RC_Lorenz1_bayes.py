# 三维数据预测的RC
#bayes参数调优 优化目标函数为平均rmse

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
# si = 0.084                    # 输入强度
# dens = 0.02                   # 储备池权重矩阵的连接密度
# leak_rate = 0.6               # 泄露率
# bias = 1.6                    # 偏置
reg = 1e-7                  #正则化参数 岭回归参数
d = 3

#生成数据集
sigma = 10
beta = 8/3
rho = 28
dt = 0.01

def lorenz(init_state,t):
    x,y,z = init_state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

t = np.arange(0,100,dt)
init_state = [1,1,1]
data = odeint(lorenz, init_state, t)
data = data.T   #转化为3*T的

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

def rc_objective(rho_Wr, si, dens, leak_rate, bias, u_train, u_test, train_size, test_size, warm):
    #初始化输入权重 W_in N*d
    W_in = np.random.uniform(-si,si,(N,d))  #输入权重
    # W_r = sp.uniform(N,N,dens)
    W_r = sp.random(
            m=N,                  # 行数
            n=N,                  # 列数
            density=dens,         # 稀疏度（非零元素比例）
            # random_state=42,     # 随机种子
            data_rvs=lambda n: np.random.uniform(-1.0, 1.0, n)  # 非零元素取值规则
        )
    # 调整W_res的谱半径（使其最大特征值的绝对值=设定的spectral_radius）
    # 谱半径=最大特征值的绝对值，这一步是为了满足回声状态特性
    # W_r = rho_Wr * W_r / max(abs(eigvals(W_r)))
    eigvals, _ = sp.linalg.eigs(W_r, k=1)  # k=1表示只计算最大的1个特征值
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
                     leak_rate * np.tanh(W_in @ input_signal + W_r @ temp_state +bias)   #形状为N*d   偏置+ bias * np.ones((N,1))
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
                     leak_rate*np.tanh(W_in @ (W_out @ temp_state) + W_r @ temp_state +bias)
        u_pred[:,t] = (W_out @ temp_state).ravel()

    # 计算每个分量的RMSE（评估x/y/z的预测精度）
    rmse_x = np.sqrt(np.mean((u_test[:, 0] - u_pred[:, 0]) ** 2))
    rmse_y = np.sqrt(np.mean((u_test[:, 1] - u_pred[:, 1]) ** 2))
    rmse_z = np.sqrt(np.mean((u_test[:, 2] - u_pred[:, 2]) ** 2))
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    return -rmse  # 负号是因为贝叶斯优化默认是最大化

def wrapped_rc_objective(bias, dens, leak_rate, rho_Wr, si):
    return rc_objective(rho_Wr, si, dens, leak_rate,
                                    bias, u_train, u_test, train_size, test_size, warm)

pbounds = {
    'rho_Wr': (0.1, 1.5),       # 储备池权重矩阵的谱半径
    'si': (0.01, 0.5),          # 输入强度
    'dens':(0.01,0.5),          # 储备池权重矩阵的连接密度
    'leak_rate': (0.1, 1.0),    # 泄露率
    'bias': (0.0, 2.0),         # 偏置
}

optimizer = BayesianOptimization(
    f = wrapped_rc_objective,
    pbounds = pbounds,
    random_state = 42,
    verbose = 2
)
optimizer.maximize(init_points=5, n_iter=20)
best_params = optimizer.max['params']
print("最优参数:", best_params)

# =============== 用最优参数再跑一次并画图 =================
rho_Wr = best_params['rho_Wr']
si = best_params['si']
dens = best_params['dens']
leak_rate = best_params['leak_rate']
bias = best_params['bias']

# 重新训练+预测
N = 200
dens = 0.02
np.random.seed(42)

W_in = np.random.uniform(-si, si, (N,d))
W_r = sp.random(
        m=N, n=N,
        density=dens,
        data_rvs=lambda n: np.random.uniform(-1.0, 1.0, n)
    )
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
                 leak_rate*np.tanh(W_in @ input_signal + W_r @ temp_state + bias)
    states[:,t] = temp_state

states = states[:, warm:]
u_train = u_train[:, warm:]
W_out = u_train @ states.T @ solve(states @ states.T + reg * np.eye(N), np.eye(states.shape[0]))

temp_state = states[:,-1]
u_pred = np.zeros((d,test_size))
for t in range(test_size):
    temp_state = (1-leak_rate)*temp_state + \
                 leak_rate*np.tanh(W_in @ (W_out @ temp_state) + W_r @ temp_state + bias)
    u_pred[:,t] = (W_out @ temp_state).ravel()

# 画图
t_test = np.arange(test_size) * dt
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
comp = ['x','y','z']
for i in range(3):
    axes[i].plot(t_test, u_test[i,:], 'b-', label=f'true_ {comp[i]}')
    axes[i].plot(t_test, u_pred[i,:], 'r--', label=f'pred_ {comp[i]}')
    axes[i].set_ylabel(f'{comp[i]} ')
    axes[i].legend()
axes[0].set_title("Lorenz (after Bayes)")
axes[2].set_xlabel('Time (s)')
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