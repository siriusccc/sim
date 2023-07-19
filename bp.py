import numpy as np
import matplotlib.pyplot as plt
import warnings
# 导入三维显示工具
from mpl_toolkits.mplot3d import Axes3D
# 导入BP模型
from sklearn.neural_network import MLPClassifier
# 导入demo数据制作方法
from sklearn.datasets import make_classification

# 制作数据
x, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=5, n_informative=3,
                           n_clusters_per_class=1, class_sep=3, random_state=10)
# 三维显示
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)     # 生成数据
ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', c=y)


# 建立BP模型
BP = MLPClassifier(solver='sgd', activation='relu', max_iter=500, alpha=1e-3, hidden_layer_sizes=(32, 32),
                   random_state=1)
# with warnings.catch_warnings():
# warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
BP.fit(x, y)

# 模型预测, 预测未知数据的标签
pred_label = BP.predict(x)
# 获取给定数据和标签的平均精度(显示预测分数)
print(BP.score(x, y))
# 可视化预测数据
print("真实类别：", x[:10])
print("预测类别：", y[:10])
plt.show()
