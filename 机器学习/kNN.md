---
title: kNN
catalog: true
date: 2019-05-17 16:40:35
subtitle: k-近邻算法
header-img:
tags: 机器学习
---

# k-近邻算法

## 概述

kNN（k-NearestNeighbor ）：又名k-近邻算法，采用不同特征值之间的距离方法进行分类。

## 特点

优点：简单，精度高，对异常值不敏感

缺点：计算复杂度高，消耗大量空间与时间

适用数据范围：数值型和标称型

补充概念：**数据范围**

**标称型：**标称型目标变量的结果只在有限目标集中取值，如真与假(标称型目标变量主要用于分类)

**数值型：**数值型目标变量则可以从无限的数值集合中取值，如0.100，42.001等 (数值型目标变量主要用于回归分析）

## 原理

我们知道样本集中每一数据与所属分类的对应关系，输入没有标签的新数据后，将新数据的每个特征与样本集中每一数据特征进行比较，然后算法提取样本集中特征最相似的数据（最近邻数据）的分类标签。一般来说，我们只选择样本数据集前 k 个最相似的数据（取多了没有什么意义，反而误差会大），通常 k 小于等于 20。

### 距离计算

![](/article/kNN/1.jpg)

### 错误率

分类器并不会得到百分百正确的结果，错误率的定义：分类器给出错误的次数除以测试执行的次数。

### python实现

```python
#_*_coding:utf-8_*_
# numpy 科学计算包
from numpy import *
# operator 运算符模块
import operator

# kNN公式
def classify0(inX, dataSet, labels, k):
    # 计算距离
    # 得到dataset的二维的长度
    dataSetSize = dataSet.shape[0]
    # tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    # 这里是因为维度不同，因此要把他们构建成同样 x * y 的矩阵才能正确的相减计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 所以这里的 sortedDistIndicies 是索引的值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 所以这里的 sortedDistIndicies 是索引值！！
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel,0)返回字典classCount中voteIlabel元素对应的值
        # 若不存在voteIlabel，则字典classCount中生成voteIlabel元素，并使其对应的数字为0，即classCount = {voteIlabel：0}
        # 当字典中有voteIlabel元素时，classCount.get(voteIlabel,0)作用是返回该元素对应的值，即0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）
    # iterable -- 可迭代对象
    # key -- 主要是用来进行比较的元素，只有一个参数，
    # 具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]
```

## 归一化数据

我们可以明显看出，计算距离中数字差值最大的属性对计算结果的影响最大，为了防止有些数据本身就很大，而导致其他属性没有那么有决定性作用，因此在处理不同取值范围的特征值时，我们通常使用数值**归一化方法**，将取值范围处理为 0 到 1 或者 -1。

### 公式

**newValue = (oldValue - min) / (max - min)**

### python实现

```python
def autoNorm(dataSet):
    # min(0) max(0) 每行最小值，每行最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # 行数
    m = dataSet.shape[0]
    # 把 1 * 3 变成 1000 * 3
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
```

