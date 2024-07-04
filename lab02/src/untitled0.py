import numpy as np
from sklearn.decomposition import TruncatedSVD

# 假设X是你的数据矩阵，它应该是一个numpy数组，其中行代表数据点，列代表特征
X = np.random.rand(100, 100)  # 这里只是一个示例矩阵

# RPCA的目标函数
def rpca_objective(X, U, sigma, lambda_):
    return np.linalg.norm(X - U @ sigma) ** 2 + lambda_ * np.linalg.norm(U, 'fro')

# RPCA的梯度
def rpca_gradient(X, U, sigma, lambda_):
    # 对U的梯度
    grad_U = 2 * (X - U @ sigma) @ sigma.T
    grad_U -= lambda_ * np.eye(U.shape[0])
    
    # 对sigma的梯度
    sigma_inv = np.linalg.inv(sigma)
    grad_sigma = -2 * U.T @ (X - U @ sigma) @ sigma_inv
    grad_sigma += lambda_ * np.kron(np.eye(U.shape[1]), sigma_inv)
    
    return grad_U, grad_sigma

# RPCA梯度下降函数
def rpca_gradient_descent(X, num_iterations, learning_rate, lambda_):
    n, m = X.shape
    U = np.random.rand(n, m)
    sigma = np.random.rand(m, m)
    
    for i in range(num_iterations):
        grad_U, grad_sigma = rpca_gradient(X, U, sigma, lambda_)
        
        # 更新U和sigma
        U -= learning_rate * grad_U
        sigma -= learning_rate * grad_sigma
        
        # 这里可以添加一个条件来停止迭代，例如梯度变化小于某个阈值
        if i % 100 == 0:
            print(f"Iteration {i}: Objective = {rpca_objective(X, U, sigma, lambda_)}")
    
    return U, sigma

# 设置参数
num_iterations = 1000
learning_rate = 0.01
lambda_ = 0.001

# 执行梯度下降
U, sigma = rpca_gradient_descent(X, num_iterations, learning_rate, lambda_)

# 使用TruncatedSVD来近似低秩部分
svd = TruncatedSVD(n_components=X.shape[1] // 2)
U_approx = svd.fit_transform(X)

# 比较近似和计算得到的低秩部分
print("Approximated low-rank matrix U:\n", U_approx)
print("Calculated low-rank matrix U:\n", U)
