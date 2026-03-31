# Yenorepo

这是一个个人的 Monorepo 架构，我给他取了一个独立的IP**Yenorepo**，用于存档平时实验过程中的各种代码，关于结构的各种事情可以参考 [README.md](./docs/architecture/README.md)


# 待解决的问题

1. 处理Cache、datasets、Outputs规范 

2. scripts是否需要common，暂时没有需要的子文件都去掉

3. 清理完毕后重新写入文件结构

4. 处理好.gitmodules、.gitignore的内容

5. 解决原有代码路径问题

6. 处理我们的复现路径问题——包含了5在内，我希望我们的项目能够运行一个sh脚本然后退回Yenomal这个工作目录，然后所有的代码就像在工作目录下进行一样，这样会方便后续开源事宜

