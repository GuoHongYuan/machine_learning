import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

experiences = np.array([1994, 1996, 1999, 2005, 2008, 2014, 2018])
salaries = np.array([325, 549, 1087, 998, 1678, 1991, 2462])

# 将特征数据集分为训练集和测试集，除了最后 4 个作为测试用例，其他都用于训练
X_train = experiences[:]
X_train = X_train.reshape(-1, 1)
X_test = experiences[:]
X_test = X_test.reshape(-1, 1)

# 把目标数据（特征对应的真实值）也分为训练集和测试集
y_train = salaries[:]
y_test = salaries[:]

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型——看就这么简单，一行搞定训练过程
regr.fit(X_train, y_train)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_test)
print('2019 Lowest is' + str(regr.predict(np.array([2019]).reshape(-1, 1))))
# 将测试结果以图标的方式显示出来
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xlabel("x")
plt.ylabel("y")
#plt.xticks(())
#plt.yticks(())

plt.show()