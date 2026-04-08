# pi05_baseline exp

## 实验目的
* pi05的baseline exp是希望保障模型的外部完备性，包括：

1. openpi模型输入datasets、ckpt正确——train正常

2. openpi输出接入rmbench正确——eval正常

## 实验过程

* 具体的实验进行过程顺畅，主要解决的问题：

1. datasets，权重切换容易出现的内存泄漏

2. 服务器openpi环境

## 实验结果

* 跑通baseline的整套流程，train-eval两个阶段

## 总结反思

* 这次在服务器上装uv环境除了好多问题：

1. av==1.14这个需要ffmpeg 7，不建议去安装ffmpeg 7，而是直接使用av的wheel（搜一下这个），一般都不建议本地编译而是去找wheel

2. 服务器上安装容易出现超时（PyPI用VPN快一些，这个问题）

## 下一步做些什么

1. 切换visual encoder正式训练，外部数据只需要处理visual encoder这里的接口（注意两边接口）

2. config使用单独的config