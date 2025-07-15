# 请确认当前的环境为base
conda activate base
# 如何创建一个新环境
conda create -n dqn_2025av python=3.9
# 如何激活创建的虚拟环境
conda activate dqn_2025av
# 安装环境需要的所有依赖前，需要进入你的工作目录
如C:\Users\73191\Desktop\dqn_2025av-main\dqn_2025av

通过 "cd+文件夹名" 打开一个文件夹

通过 "cd .." 返回上一级文件夹
# 进入工作目录后，可以安装环境所需的所有依赖
rl-agents(算法实现的库)和highway_env(自动驾驶环境)的所有依赖已经置于
dqn_2025av-main\dqn_2025av\requirements.txt下

所以你只需运行指令

pip install -r requirements.txt

即可安装所有环境依赖，之后即可在自动驾驶环境中训练或测试一个智能体
# 如何运行基于规则的车辆(自车)
由于此时车辆所有的行为由规则定义，所以不需要训练，运行指令

python experiments.py evaluate configs/MergeEnv/env_rule.json --test --episodes=20

即可运行
# 如何训练一个智能体
一般结构：python experiments.py evaluate 环境配置 智能体配置 --train --no-display --episodes=20

具体实例

python experiments.py evaluate configs/MergeEnv/env_agg.json configs/MergeEnv/agents/DQNAgent/dqn.json --train --no-display --episodes=20

"--no-display"参数传入(设置为True)意味着不会保存渲染视频。如果训练期间需要运行几万个episodes，出于内存考虑，此参数训练期间必须传入
# 如何测试一个训练好的智能体 
python experiments.py evaluate configs/MergeEnv/env_agg.json configs/MergeEnv/agents/DQNAgent/dqn.json --test --recover --episodes=20

此时需要看到策略的效果，默认不传入"--no-display"，以保存测试时的渲染视频

"--recover"参数意味着加载最新一次训练的模型

如果你不想加载最新的模型，需要在"--recover"后传入具体的模型路径

如 --recover out/MergeEnv/DQNAgent/run_20250617-145431_621696/checkpoint-final.tar
# 如何查看车辆运行的渲染视频
所有运行结果,包括训练的模型和已经渲染的视频会保存在工作目录的out文件夹下
# 如何查看模型的训练表现(使用tensorboard)
在requirement.txt中已经安装tensorboard

你只需要找到out文件夹下当次运行的目录，利用指令

tensorboard --logdir=/data/code/xiongminghao/rl-agents/scripts/out/MergeEnv/DQNAgent/run_20250617-145431_621696

即可查看智能体训练获得的奖励情况



