import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        """阶跃激活函数"""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """训练感知机"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 计算加权输入
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_func(linear_output)

                # 计算权重更新量
                update = self.lr * (y[idx] - y_predicted)

                # 更新权重和偏置
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """预测输出"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step_func(linear_output)

# 训练 AND 门感知机
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 输入
y_and = np.array([0, 0, 0, 1])  # AND 逻辑表输出

perceptron_and = Perceptron(learning_rate=0.1, n_iters=1000)
perceptron_and.fit(X_and, y_and)

# 测试 AND 门
print("AND result:")
print(perceptron_and.predict(X_and))  # 预期输出: [0, 0, 0, 1]

# 训练 OR 门感知机
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 输入
y_or = np.array([0, 1, 1, 1])  # OR 逻辑表输出

perceptron_or = Perceptron(learning_rate=0.1, n_iters=1000)
perceptron_or.fit(X_or, y_or)

# 测试 OR 门
print("OR result:")
print(perceptron_or.predict(X_or))  # 预期输出: [0, 1, 1, 1]
