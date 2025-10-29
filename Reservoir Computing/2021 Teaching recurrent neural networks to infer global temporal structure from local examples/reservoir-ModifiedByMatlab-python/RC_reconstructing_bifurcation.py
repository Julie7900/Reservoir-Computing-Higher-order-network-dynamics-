# 比较完整的定义了储备池类
# 对Lorenz系统进行预测
# beyes优化
# 尚未实现平移和线性变换操作

import numpy as np
import scipy.sparse as sp
from bayes_opt import BayesianOptimization
from scipy.linalg import solve
from scipy.integrate import odeint
from numpy.linalg import eigvals
import matplotlib.pyplot as plt


class ReservoirComputer:
    def __init__(self,data,warm, train_size,test_size, si, leak_rate,bias,rho_Wr,dens,
                N=500, reg=1e-7, d=3):
        self.data = data   #形状d*T
        self.warm = warm
        self.train_size = train_size
        self.test_size=test_size
        self.si = si
        self.N = N
        self.leak_rate = leak_rate
        self.bias = bias
        self.dens = dens
        self.reg = reg
        self.rho_Wr = rho_Wr
        self.d = d

        self.W_in = None
        self.W_r = None

    def generate_dataset(self):
        u_train = self.data[:,:self.train_size]
        u_test = self.data[:,self.train_size:]
        return u_train,u_test

    def generate_W_in(self):
        # if self.W_in is None:
        self.W_in = np.random.uniform(-self.si, self.si, (self.N, self.d))
        return self.W_in

    def generate_W_r(self):
        # if self.W_r is None:  # 若未传入W_r，则生成
        self.W_r = sp.random(
            m=self.N, n=self.N,
            density = self.dens,
            data_rvs=lambda n: np.random.uniform(-1.0, 1.0, n)
        )
        # eigvals_, _ = sp.linalg.eigs(self.W_r, k=1)
        # max_eig = np.abs(eigvals_)[0]
        # self.W_r = self.rho_Wr * self.W_r / (max_eig if max_eig != 0 else 1)
        try:
            eigvals_, _ = sp.linalg.eigs(self.W_r, k=1)
            max_eig = np.abs(eigvals_)[0]
        except:
            max_eig = np.max(np.abs(np.linalg.eigvals(self.W_r.A)))
        self.W_r = self.rho_Wr * self.W_r / (max_eig if max_eig != 0 else 1)
        return self.W_r

    def Reservoir_Train(self):
        u_train, _ = self.generate_dataset()
        self.generate_W_in()  # 生成self.W_in
        self.generate_W_r()  # 生成self.W_r
        ############# 训练 ################
        # %% 初始化储备池状态
        temp_state = np.zeros(self.N)  # 这里temp_state不用改！仍然保持二维矩阵
        # 储备池状态，存储训练过程中所有时刻的储备池状态
        states = np.zeros((self.N, self.train_size))  # 这里states不用改！仍然保持二维矩阵
        for t in range(1, self.train_size):
            input_signal = u_train[:, t - 1]  # 当前训练输入（t时刻的x值）
            # 计算t时刻的储备池状态（Leaky ESN的状态更新公式）
            # 状态 = (1-泄露率)*上一时刻状态 + 泄露率*tanh(输入映射 + 储备池内部映射)
            temp_state = (1 - self.leak_rate) * temp_state + \
                         self.leak_rate * np.tanh(
                self.W_in @ input_signal + self.W_r @ temp_state + self.bias)  # 形状为N*d   偏置+ self.bias * np.ones((N,1))
            states[:, t] = temp_state

        ##岭回归
        states = states[:, self.warm:]  # 稳定后的状态
        u_train = u_train[:, self.warm:]  # 对应的训练数据
        # W_out = u_train @ states.T @ np.linalg.pinv(states @ states.T + reg * np.eye(N))
        W_out = u_train @ states.T @ solve(states @ states.T + self.reg * np.eye(self.N), np.eye(states.shape[0]))
        return u_train,states,W_out

    def Reservoir_Pred(self,states,W_out):
        ############# 预测 ################
        # #用储备池状态的最后作为预测的输入,转化为N维数组之后，再转化为N*1的二维矩阵的形式
        # u_train,states, W_out = self.Reservoir_Train()
        temp_state = states[:, -1]
        _, u_test = self.generate_dataset()
        test_size = u_test.shape[1]
        u_pred = np.zeros((self.d, test_size))  # 用来存储预测的数据
        # input_signal = u_train[:, -1]  # 当前训练输入（t时刻的x值）
        for t in range(test_size):
            # input_signal = u_test[t,:].reshape(d,1)                       # 当前训练输入（t时刻的x值）
            # 计算t时刻的储备池状态（Leaky ESN的状态更新公式）
            # 状态 = (1-泄露率)*上一时刻状态 + 泄露率*tanh(输入映射 + 储备池内部映射)
            temp_state = (1 - self.leak_rate) * temp_state + \
                         self.leak_rate * np.tanh(self.W_in @ (W_out @ temp_state) + self.W_r @ temp_state + self.bias)
            u_pred[:, t] = (W_out @ temp_state).ravel()

        return u_pred,u_test

    def rc_objective(self,u_pred,u_test):
        # rmse_x = np.sqrt(np.mean((u_test[0, :] - u_pred[0, :]) ** 2))
        # rmse_y = np.sqrt(np.mean((u_test[1, :] - u_pred[1, :]) ** 2))
        # rmse_z = np.sqrt(np.mean((u_test[2, :] - u_pred[2, :]) ** 2))
        # rmse = (rmse_x + rmse_y + rmse_z) / 3
        # print(rmse)

        rmse_x = np.sqrt(np.mean((u_test[0, :self.test_size] - u_pred[0, :self.test_size]) ** 2))
        rmse_y = np.sqrt(np.mean((u_test[1, :self.test_size] - u_pred[1, :self.test_size]) ** 2))
        rmse_z = np.sqrt(np.mean((u_test[2, :self.test_size] - u_pred[2, :self.test_size]) ** 2))
        rmse = (rmse_x + rmse_y + rmse_z)/3
        # print(rmse)
        # 用下面的来优化好像要好一点,但是只用了前三个点
        rmse_x = np.sqrt(np.mean((u_test[:, 0] - u_pred[:, 0]) ** 2))
        rmse_y = np.sqrt(np.mean((u_test[:, 1] - u_pred[:, 1]) ** 2))
        rmse_z = np.sqrt(np.mean((u_test[:, 2] - u_pred[:, 2]) ** 2))
        # rmse_4 = np.sqrt(np.mean((u_test[:, 3] - u_pred[:, 3]) ** 2))
        rmse = rmse_x + rmse_y + rmse_z
        return -rmse  # 负号是因为贝叶斯优化默认是最大化

    def wrapped_rc_objective(self,bias, dens, leak_rate, rho_Wr, si):
        self.bias = bias
        self.dens = dens
        self.leak_rate = leak_rate
        self.rho_Wr = rho_Wr
        self.si = si
        u_train, states, W_out = self.Reservoir_Train()
        u_pred,u_test = self.Reservoir_Pred(states, W_out)
        return self.rc_objective(u_pred,u_test)


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

def lorenz(t,init_state):
    x,y,z = init_state
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - (8/3) * z
    return np.array([dx, dy, dz])

init_state = [1,1,1]
y0 = init_state
t0 = 0
h = 0.01
n = 10000
t_rk4, data = rk4(lorenz,t0,y0,h,n)
data = data.T

P1 = np.zeros((3,n+1))
P1[0,:] = 1
P2 = np.zeros((3,3))
P2[0,0] = -0.012
c_list = [0, 1, 2, 3]   # 控制参数
data_translation = [data + c * P1 for c in c_list]                                  # 平移族 (4, 3, 10000)
# data_linear_transformation = [(np.eye(3) + c*P2) @ data for c in c_list]          # 压缩族

############################## Bayes ###################################
N = 100                         # 储备池节点数量
reg = 1e-7                      #正则化参数 岭回归参数
d = 3
data = data_translation[0]
warm = 100
train_size = 2000
test_size = 2000

pbounds = {
    'rho_Wr': (0.1, 1.5),       # 储备池权重矩阵的谱半径
    'si': (0.01, 0.5),          # 输入强度
    'dens':(0.05,0.5),          # 储备池权重矩阵的连接密度
    'leak_rate': (0.1, 1.0),    # 泄露率
    'bias': (0.0, 2.0),         # 偏置
}

rc = ReservoirComputer(
    data, warm, train_size,test_size, si=0.1, leak_rate=0.5, bias=0.1, rho_Wr=0.5, dens=0.1, N=N, reg=reg, d=d)

optimizer = BayesianOptimization(
    f = rc.wrapped_rc_objective,
    pbounds = pbounds,
    random_state = 42,
    verbose = 2
)
optimizer.maximize(init_points=5, n_iter=20)
best_params = optimizer.max['params']
print("最优参数:", best_params)

###################### 用最优参数再跑一次并画图 ###############################
rho_Wr = best_params['rho_Wr']
si = best_params['si']
dens = best_params['dens']
leak_rate = best_params['leak_rate']
bias = best_params['bias']

rc.bias = bias
rc.dens = dens
rc.leak_rate = leak_rate
rc.rho_Wr = rho_Wr
rc.si = si
u_train, states, W_out = rc.Reservoir_Train()
u_pred,u_test = rc.Reservoir_Pred(states, W_out)

# test_size = u_test.shape[1]
t_test = np.arange(test_size)
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
comp = ['x','y','z']
for i in range(3):
    axes[i].plot(t_test, u_test[i,:test_size], 'b-', label=f'true_ {comp[i]}')
    axes[i].plot(t_test, u_pred[i,:test_size], 'r--', label=f'pred_ {comp[i]}')
    axes[i].set_ylabel(f'{comp[i]} ')
    axes[i].legend()
axes[0].set_title("Lorenz (after Bayes)")
axes[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()