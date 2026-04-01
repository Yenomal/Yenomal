请在执行过程中满足以下约束：

1. 最终生成的bash运行脚本（包括train，eval）需要能够最简运行，所有的参数全部在bash脚本内部设定，配备默认值，用户使用时外部只需要在bash脚本内部配置参数，然后运行./xx.sh即可

2. train过程需要可视化，使用wandb，记录loss，grad_norm（此处grad_norm记录xxxxxxxxxx）

3. 输出一个run.md文件，告诉我如何运行train，eval脚本，注明可以修改的参数，使用短句表示其作用

4. sh脚本保留在根目录，python脚本保留在对应的exp文件夹下(此处为xxxxxxxxxx)