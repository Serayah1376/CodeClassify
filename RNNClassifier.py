# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:00:58 2021

@author: 10983
"""
import torch

USE_GPU=True


class RNNClassifier(torch.nn.Module):
    '''
    input_size:N_CHARS
    hidden_size:HIDDEN_SIZE
    output_size:N_LABEL
    n_layers=N_LAYER
    '''
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=False):
        super(RNNClassifier,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1 #双向2 单向1
        self.bidirectional=bidirectional
        
        
    def _init_hidden(self,batch_size):
        #初始化全零隐层
        hidden=torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return self.create_tensor(hidden)  #create_tensor的作用就是判断是否将数据放在显卡上
    
    #seq_lengths为该样本长度
    def forward(self,input):
        #input shape: B x S -> S x B
        #print('*****inputsize=',input.size()) [16,229]
        input = input.t()  #做一个转置
        batch_size=input.size(1)
        
        hidden=self._init_hidden(batch_size)
        embedding=self.embedding(input) 
        
        #加快速度
        '''
        要求input中的样本是按照长度是有序的
        '''
        # 将一个 填充过的变长序列 压紧  可以不用
        #gru_input=torch.nn.utils.rnn.pack_padded_sequence(embedding,seq_lengths)
        
        output,hidden=self.gru(embedding,hidden) #如果是双向的话hidden里就有两个对象
        if self.n_directions==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)#如果两层的话就拼接起来
        else:
            hidden_cat=hidden[-1]
        fc1_output=self.fc1(hidden_cat)
        fc_output=self.fc2(fc1_output)
        return fc_output  #[4,104]
    
    #模型创建
    def build(self):
        #embedding层输出维度hidden_size即为gru层的输入维度
        self.embedding=torch.nn.Embedding(self.input_size,self.hidden_size) 
        #GRU层
        self.gru=torch.nn.GRU(self.hidden_size,self.hidden_size,self.n_layers,bidirectional=self.bidirectional)
        #线性层
        self.fc1=torch.nn.Linear(self.hidden_size*self.n_directions, 160)
        self.fc2=torch.nn.Linear(160,104)

    #判断是否放到显卡上
    def create_tensor(self,t):
        if USE_GPU:
            device=torch.device('cuda:0')
            t=t.to(device)
        return t 
    

            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
          

 
        