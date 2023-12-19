# CTD3

所有被全部注释的文件和test文件夹下的文件都忽略

## Train
run the training script file:
`python train_acc(td3_mine).py`
or
`python train_lanekeeping_test.py`
会在根目录下生成训练时的数据，以及projects下获取训练时的附加信息 如网络参数 损失、奖励值变化

parse_args定义了几个命令行参数 不大重要 可以看一眼

## Deploy

`python deploy_acc.py` 
用训练好的网络生成小车运行轨迹

## Environment
highway_env文件夹下

## Abstract algo
两个阶段

第一阶段是/tool/Yida/tools.py 对训练数据（日志文件）作区间化处理 生成csv文件
目前输入文件是死的 后面会改

第二阶段是data_analysis/acc_td3/mdp/construct_mdp.py(构建mdp模型)
和data_analysis/acc_td3/mdp/my_kmeans.py(用mdp模型聚类)   
后者会报错，还没写出来

## Evaluation
data_analysis/acc_td3/plt_diagram 用canopy、gap_statistic、elbow、轮廓系数四种方法
确定聚类中心数 并画图和计算误差 模型和图在data_analysis/acc_td3下的imgs和mdls文件夹

针对的是原始kmeans方法 用来和新算法对比的

有一些命令行参数 可能对运行结果有影响，可以多留意一下

canopy和gap_statistic实现在data_analysis下 计算误差时要额外用到data_analysis下的
eval_acc.py
