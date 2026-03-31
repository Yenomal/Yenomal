# config的要求

config 每一个配置都需要注释！！！

## src/common

众多的超参数调节要求尽量通过config暴露出来，包括各种接口配置

## src/projects

配置各个实验下的相关配置

## scripts/projects

1. scripts不能包含任何底层的运行脚本，比如datasets的meta、norm等，只保留复现时的入口

2. bash脚本尽量避免CLI输入，而是在bash内完成配置，故应该尽量减少 NUM_WORKERS="${NUM_WORKERS:-4}" 此等类型，而是直接在python运行的命令下建议用户输入，并给出提示信息

