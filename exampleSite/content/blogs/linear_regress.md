---
title: "Chapter1 线性回归"
date: 2025-07-05T21:30:45+08:00
draft: false
author: "Jiamin Jiang"
tags:
  - Deep Learing
image: /images/d2l.png
description: "这是一个李沐课程《动手学深度学习》的一章节"
toc: true
mathjax: true
---
# Chapter1 线性回归

问题定义：给定一个数据集，我们的目标是寻找模型的权重 \\( \mathbf{w} \\) 和偏置 \\(b\\)。

当我们的输入包含 \\( d \\) 个特征时，我们将预测结果 \\( \hat{y} \\)，（通常使用“尖角”符号表示 \\( y \\) 的估计值）表示为：

$$ \hat{y} = w_1  x_1 + ... + w_d  x_d + b $$

将所有特征放到向量 \\( \mathbf{x} \in \mathbb{R}^d \\) 中，并将所有权重放到向量 \\( \mathbf{w} \in \mathbb{R}^d \\) 中，通过点积可以表示模型：

$$ \hat{y} = \mathbf{w}^\top \mathbf{x} + b $$

向量 \\( \mathbf{x} \\) 对应于单个数据样本的特征。进一步，我们考虑多个样本点的情况。用符号表示的矩阵 \\( \mathbf{X} \in \mathbb{R}^{n \times d} \\)，可以很方便地引用我们整个数据集的 \\( n \\) 个样本。其中，\\( \mathbf{X} \\) 的每一行是一个样本，每一列是一种特征。

对于特征集合 \\( \mathbf{X} \\)，预测值 \\( \hat{\mathbf{y}} \in \mathbb{R}^n \\) 可以通过矩阵-向量乘法表示为：

$$ \hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b $$

给定训练数据特征 \\( \mathbf{X} \\) 和对应的已知标签 \\( \mathbf{y} \\)，线性回归的目标是找到一组权重向量 \\( \mathbf{w} \\) 和偏置 \\( b \\) 使得新样本预测标签的误差尽可能小。

## 损失函数（Loss function）

回归问题中最常用的损失函数是平方误差函数，当样本 \\( i \\) 的预测值为 \\( \hat{y}^{(i)} \\)，其相应的真实标签为 \\( y^{(i)} \\) 时，平方误差可以定义为以下公式：

$$ l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 $$

常数 \\( \frac{1}{2} \\) 不会带来本质的差别，但这样在形式上稍微简单一些（因为当我们对损失函数求导后常数系数为 1），带入 \\( \hat{\mathbf{y}} \\) 求平均得到：

$$ L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2 $$

在训练模型时，我们希望寻找一组参数 \\( \mathbf{w}^*, b^* \\)，这组参数能最小化在所有训练样本上的总损失。如下式：

$$ \mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\ L(\mathbf{w}, b) $$

## 解析解

线性回归的解可以用一个公式简单地表达出来，这类解叫做解析解（analytical solution）。首先，我们将偏置 \\( b \\) 合并到参数 \\( \mathbf{w} \\) 中，合并方法是在包含所有参数的矩阵中附加一列。

我们的预测问题是最小化 \\( \ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{2 n}\|\mathbf{y}-\mathbf{X w}\|^{2} \\)。我们对其求偏导得到：

$$ \frac{\partial}{\partial \mathbf{w}} \ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}(\mathbf{y}-\mathbf{X w})^{T} \mathbf{X} $$

损失函数是凸函数，因此在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。将损失关于 \\( \mathbf{w} \\) 的导数设为 0，得到解析解：

$$
\begin{align} 
& \frac{\partial}{\partial \mathbf{w}} \ell(\mathbf{X}, \mathbf{y}, \mathbf{w})=0 \\
&\Leftrightarrow   \frac{1}{n}(\mathbf{y}-\mathbf{X w})^{T} \mathbf{X}= 0\\
&\Leftrightarrow \mathbf{w}^* = (\mathbf{X }^{T}\mathbf{X })^{-1}\mathbf{X }\mathbf{y}
\end{align}
$$

> 💡 像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。

## 总结：

- 线性回归是对 n 维输入的加权，外加偏差  
- 使用平方损失来衡量预测值和真实值的差异  
- 线性回归有显示解  
- 线性回归可以看作是单层神经网络  

## 基础优化算法

如何快速地求出 \\( \mathbf{w}^*, b^* \\) 的一些算法

### 梯度下降

梯度下降（gradient descent）方法几乎可以优化所有深度学习模型。它通过不断地在损失函数递减的方向上更新参数来减低误差。

梯度下降的步骤：

1. 挑选一个初始值 \\( \mathbf{w}_0 \\)
2. 重复迭代参数 \\( t =1,2,3,\cdots \\)

$$ \mathbf{w}_{t}=\mathbf{w}_{t-1}-\eta \frac{\partial \ell}{\partial \mathbf{w}_{t-1}} $$

- 沿梯度方向是增加损失函数 \\( \ell(\mathbf{X}, \mathbf{y}, \mathbf{w}) \\) 的值，因此在导数前面加了负
