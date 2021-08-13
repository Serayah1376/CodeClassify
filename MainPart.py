# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:28:45 2021

@author: 10983
"""

import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import Preparing_Data as PD
import RNNClassifier as RNN
#parameters
HIDDEN_SIZE=256
BATCH_SIZE=16 #改为8试一下
N_LAYER=2  
N_EPOCHS=100
USE_GPU=True #使用GPU
gamma=0.1
mini_loss=100 #最小损失的初始值

trainset=PD.CodeDataset(is_train_set=True)
testset =PD.CodeDataset(is_train_set=False)

#将分词后的codes进行padding：取每个batch的列表中的最大长度作为填充目标长度
def make_tensors(batch,is_train=True):
    codes=[item[0] for item in batch]
    labels=[item[1] for item in batch]
    list2=[]
    maxlen=trainset.getMaxlen(codes)
    for s in codes:
        if is_train:
            list1=[trainset.word2index[word] for word in s]
        else:
            list1=[testset.word2index[word] for word in s]
        #padding:填充0是长度相等
        i=0
        need_add=maxlen-len(list1)
        while i<need_add:
            list1.append(0)
            i+=1
        list2.append(list1)
    #将代码向量转化为张量
    seq_tensor=torch.LongTensor(list2)
    labels=torch.LongTensor(labels)     
    return trainset.create_tensor(seq_tensor),\
          trainset.create_tensor(labels)

#数据的准备
#训练集
trainset=PD.CodeDataset(is_train_set=True)
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=make_tensors)
#验证集
validset =PD.CodeDataset(is_train_set=False)
validloader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=lambda x: make_tensors(x, False))

#得到标签的种类数，决定模型最终输出维度的大小
#训练集中的类别是最全的，故使用训练集的label_num
N_LABEL=trainset.getLabelNum() 
#词典中词的个数，在使用词嵌入Embedding的时候传入的参数
N_WORDS_train=trainset.dicnum 
  
#N_WORDS_test=testset.dicnum   
#计算运行的时间，转换分钟和秒的形式
def time_since(since):
    s=time.time()-since
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds'% (m,s)

#训练过程
def trainModel(model1):
    total_loss=0
    for i,(inputs,target) in enumerate(trainloader,1):
        output=model1.forward(inputs)
        loss=criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        total_loss+=loss.item()
        if i%100==0:
            print(f'[{time_since(start)}] Epoch {epoch}',end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]',end='')
            print(f'loss={total_loss/(i*len(inputs))}')
    #如果出现损失值的最新低值，则保存网络模型
    if (total_loss/len(trainset))<mini_loss:
        print('*************')
        torch.save(model1.state_dict(),'C:/Users/10983/py入门/GRUClassifier/net_params.pkl')
    return total_loss  

def validModel(model2):
    correct=0
    total=len(testset)
    test_total_loss=0
    print('evaluating trained model...')
    model2.load_state_dict(torch.load('C:/Users/10983/py入门/GRUClassifier/net_params.pkl'))
    with torch.no_grad():#测试不需要求梯度
        for i,(inputs,target) in enumerate(validloader,1):
            output=model2.forward(inputs)
            test_loss=criterion(output,target)
            pred=output.max(dim=1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
            
            test_total_loss+=test_loss.item()
            if i%100==0:
                print(f'[{time_since(start)}] Epoch {epoch}',end='')
                print(f'[{i * len(inputs)}/{len(testset)}]',end='')
                print(f'test_loss={test_total_loss/(i*len(inputs))}')
           
        percent='%.2f' % (100 * correct /total)
        print(f'Test set: Accuracy {correct}/{total} {percent} %')
        
    return correct / total  
 
'''
N_CHARS：整个字母表元素个数
HIDDEN_SIZE：隐层维度
N_LABEL：有多少个分类
N_LAYER：用几层
'''
classifier=RNN.RNNClassifier(N_WORDS_train,HIDDEN_SIZE,N_LABEL,n_layers=N_LAYER)
classifier.build()
#是否用GPU
if USE_GPU:
    device=torch.device('cuda:0')
    classifier.to(device)
    
#交叉熵损失
criterion=torch.nn.CrossEntropyLoss()
#优化器
optimizer=torch.optim.Adam(classifier.parameters(),lr=0.1)
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
#看训练的时间有多长
start=time.time()
print('Training for %d epochs...' % N_EPOCHS)
#存放训练的结果
acc_list=[]
for epoch in range(1,N_EPOCHS+1):
    #print('%d'%epoch,':',end='')
    trainModel(classifier)  #############
    acc=validModel(classifier)
    acc_list.append(acc) 
    
#绘图部分看准确率的变化
epoch=np.arange(1,len(acc_list)+1,1)
acc_list=np.array(acc_list)
plt.plot(epoch,acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
