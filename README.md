# CodeClassify
对C/C++代码的分类
# 详细功能
不同形式的代码可能实现的是相同的功能，该项目的任务就是将实现相同功能的不同形式代码分为一类
# 环境
python 3.7  
pytorch 1.9.0   
# 代码介绍  
* MainPart.py：要运行的文件，进行参数的初始化，进行训练，验证和测试
* Preparing_Data.py：进行数据的预处理
* RNNClassifier.py：神经网络的构建
* train.py：训练，验证和测试部分
# 运行情况
* 第10版及之前的提交；对数据的处理类似于普通英文文本的预处理，网络用的是GRU，40个epoch运行下来，在验证集上的准确率为95.38%，测试集上的准确率为95.19%
