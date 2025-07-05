# Chapter1 线性回归

问题定义：给定一个数据集，我们的目标是寻找模型的权重$\mathbf{w}$和偏置$b$。

当我们的输入包含 $d$ 个特征时，我们将预测结果 $\hat{y}$，（通常使用“尖角”符号表示 $y$ 的估计值）表示为：

$$ \hat{y} = w_1  x_1 + ... + w_d  x_d + b $$

将所有特征放到向量  $\mathbf{x} \in \mathbb{R}^d$ 中，并将所有权重放到向量 $\mathbf{w} \in \mathbb{R}^d$ 中，通过点积可以表示模型：

$$ \hat{y} = \mathbf{w}^\top \mathbf{x} + b $$

向量$\mathbf{x}$对应于单个数据样本的特征。进一步，我们考虑多个样本点的情况。用符号表示的矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$，可以很方便地引用我们整个数据集的$n$个样本。其中，$\mathbf{X}$的每一行是一个样本，每一列是一种特征。

对于特征集合$\mathbf{X}$，预测值$\hat{\mathbf{y}} \in \mathbb{R}^n$可以通过矩阵-向量乘法表示为：

$$ {\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b $$

给定训练数据特征$\mathbf{X}$和对应的已知标签$\mathbf{y}$，线性回归的目标是找到一组权重向量$\mathbf{w}$和偏置$b$使得新样本预测标签的误差尽可能小。

# 损失函数（Loss function）

回归问题中最常用的损失函数是平方误差函数，当样本$i$的预测值为$\hat{y}^{(i)}$，其相应的真实标签为$y^{(i)}$时，平方误差可以定义为以下公式：

$$ l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 $$

常数$\frac{1}{2}$不会带来本质的差别，但这样在形式上稍微简单一些（因为当我们对损失函数求导后常数系数为$1$），带入${\hat{\mathbf{y}}}$求平均得到：

$$ L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2 $$

在训练模型时，我们希望寻找一组参数（$\mathbf{w}^*, b^*$），这组参数能最小化在所有训练样本上的总损失。如下式：解析解

$$
\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\ L(\mathbf{w}, b)
$$

# 解析解

线性回归的解可以用一个公式简单地表达出来，这类解叫做解析解（analytical solution）。首先，我们将偏置$b$合并到参数$\mathbf{w}$中，合并方法是在包含所有参数的矩阵中附加一列。

我们的预测问题是最小化 $\ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{2 n}\|\mathbf{y}-\mathbf{X w}\|^{2}$。我们对其求偏导得到：

$$
 \frac{\partial}{\partial \mathbf{w}} \ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}(\mathbf{y}-\mathbf{X w})^{T} \mathbf{X}
$$

损失函数是凸函数，因此在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。将损失关于$\mathbf{w}$的导数设为0，得到解析解：

$$
\begin{align} 
& \frac{\partial}{\partial \mathbf{w}} \ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=0 \\
&\Leftrightarrow   \frac{1}{n}(\mathbf{y}-\mathbf{X w})^{T} \mathbf{X}= 0\\
&\Leftrightarrow \mathbf{w}^* = (\mathbf{X }^{T}\mathbf{X })^{-1}\mathbf{X }\mathbf{y}
\end{align}
$$

<aside>
💡

像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。 解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。

</aside>

总结：

- 线性回归是对 n 维输入的加权，外加偏差
- 使用平方损失来衡量预测值和真实值的差异
- 线性回归有显示解
- 线性回归可以看作是单层神经网络

# 基础优化算法

如何快速地求出（$\mathbf{w}^*, b^*$）的一些算法

## 梯度下降

梯度下降（gradient descent）方法几乎可以优化所有深度学习模型。它通过不断地在损失函数递减的方向上更新参数来减低误差。

> 梯度下降最简单的方法是计算损失函数（数据集中所有样本的损失均值）关于模型参数的偏导数（又称为梯度）。
> 

![蓝色是损失函数的等高线](Chapter1%20%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%202054fd39d708806a97a8ed3bb251792e/image.png)

蓝色是损失函数的等高线

梯度下降的步骤：

1. 挑选一个初始值 $\mathbf{w}_0$
2. 重复迭代参数 $t =1,2,3,\cdots$
    
    $$
    \mathbf{w}_{t}=\mathbf{w}_{t-1}-\eta \frac{\partial \ell}{\partial \mathbf{w}_{t-1}}
    $$
    
    - 沿梯度方向是增加损失函数$\ell(\mathbf{X}, \mathbf{y}, \mathbf{w})$的值，因此在导数前面加了负号；
    - 学习率$\eta$：步长的超参数。学习率不能太小，也不能太大。参考：[1.3 学习率 $\alpha$](https://www.notion.so/1-3-alpha-2004fd39d7088045a74bd83af5e41568?pvs=21)

## 小批量随机梯度下降

在实际中，每次计算 $\mathbf{w}$ 的梯度都必须遍历整个数据集。因此，我们通常会在每次需要计算更新的时候随机抽取一小批样本， 这种变体叫做***小批量随机梯度下降***（minibatch stochastic gradient descent）。

在每次迭代中，我们首先随机抽样一个小批量 $\mathcal{B}$，它是由固定数量的训练样本组成的。然后，我们计算小批量的平均损失关于模型参数的导数（也可以称为梯度）。最后，我们将梯度乘以一个预先确定的正数$\eta$，并从当前参数的值中减掉。我们用下面的数学公式来表示这一更新过程（$\partial$表示偏导数）：

$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
$$

总结一下，算法的步骤如下： 

1. 初始化模型参数的值，如随机初始化； 
2. 从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤。 对于平方损失和仿射变换，我们可以明确地写成如下形式:
    
    $$
    \begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}
    $$
    

其中， $\mathbf{w}$  和  $\mathbf{x}$  都是向量。${|\mathcal{B}|}$ 表示每个小批量中的样本数，这也成为批量大小（batch size）。$\eta$ 表示学习率（learning rate）。批量大小和学习率的值通常是手动预先指定，而不是通过模型训练得到的。这些可以调整但不在训练过程中更新的参数称为*超参数*（hyperparameter）。*调参*（hyperparameter tuning）是选择超参数的过程。超参数通常是我们根据训练迭代结果来调整的，而训练迭代结果是在独立的*验证数据集*（validation dataset）上评估得到的。

在训练了预先确定的若干迭代次数后（或者直到满足某些其他停止条件后），我们记录下模型参数的估计值，表示为$\hat{\mathbf{w}}, \hat{b}$。但是，即使我们的函数确实是线性的且无噪声，这些估计值也不会使损失函数真正地达到最小值。因为算法会使得损失向最小值缓慢收敛，但却不能在有限的步数内非常精确地达到最小值。

## 代码实现1

```python
# 线性回归——从零开始实现
import random
import torch 
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成y = Xw + b + 噪声"""

    # 生成均值=0,方差=1,样本数为{num_examples},特证数=w的长度的一组随机样本
    X = torch.normal(0, 1, (num_examples, len(w))) 
    # y = Xw + b
    y = torch.matmul(X, w) + b
    # 加入随机噪声
    y += torch.normal(0, 0.01, y.shape)
    # 把X,和y做成一个列向量返回
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabels:', labels[0])

# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(),
#                 labels.detach().numpy(), 1);

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):    
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
 
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法
"""
    inputs: 
        params: w, b
        lr: learning rate
        batch_size: the number of sample
"""
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练过程
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # step1: forward pass
        l = loss(net(X, w, b), y)
        # 因为 l 的形状是 batch_size*1, 而不是一个标量
        # back propagation
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}') 
```

## 代码实现2（简洁版本）

```python
# 线性回归的简洁实现
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""
将features和labels作为API的参数传递,
并通过数据迭代器指定batch_size。
布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。"""
def load_array(data_arrays, batch_size, is_train=True):
    """ 构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# step1: 定义模型
# 思路: 首先定义一个模型变量 net，它是一个 Sequential 类的实例。Sequential 类将多个层串联在一起。
# 当给定输入数据时，Sequential 实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，以此类推

# 在PyTorch中，全连接层在Linear类中定义。 
# 值得注意的是，我们将两个参数传递到nn.Linear中。 
# 第一个指定输入特征形状，即2，x = [x1, x2]
# 第二个指定输出特征形状，输出特征形状为单个标量，因此为1。y = [y1]
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

# step2: 初始化模型参数
# 在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 
# 偏置参数将初始化为零。

# 通过 net[0] 选择网络中的第一个图层
# 使用weight.data和bias.data方法访问参数。 
# 使用替换方法normal_和fill_来重写参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# print(net[0].weight.data, net[0].bias.data)

# step3: 定义损失函数
# [计算均方误差使用的是MSELoss类，也称为平方L2范数]。
# 默认情况下，它返回所有样本损失的平均值
loss = nn.MSELoss()

# step4: 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# step5: 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 比较训练后的[w, b]与标准的[w, b]
w = net[0].weight.data
print("w 的估计误差: ", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("b 的估计误差: ", true_b - b)
```