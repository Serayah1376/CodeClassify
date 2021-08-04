# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:00:58 2021

@author: 10983
"""
import torch

USE_GPU=True

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        super(RNNClassifier,self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1 #双向2 单向1
        
        self.embedding=torch.nn.Embedding(input_size,hidden_size)
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        
        self.fc=torch.nn.Linear(hidden_size*self.n_directions, output_size)
       
    def _init_hidden(self,batch_size):
        #初始化全零隐层
        hidden=torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return self.create_tensor(hidden)  #create_tensor的作用就是判断是否将数据放在显卡上
    
    
    #seq_lengths为该样本长度
    def forward(self,input,seq_lengths):
        #input shape: B x S -> S x B
        input = input.t()  #做一个转置
        batch_size=input.size(1)
        
        hidden=self._init_hidden(batch_size)
        embedding=self.embedding(input) 
        
        #加快速度
        '''
        要求input中的样本是按照长度是有序的
        '''
        # 将一个 填充过的变长序列 压紧
        gru_input=torch.nn.utils.rnn.pack_padded_sequence(embedding,seq_lengths)#???????一定要在cpu上吗
        
        output,hidden=self.gru(gru_input,hidden) #如果是双向的话hidden里就有两个对象
        if self.n_directions==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)#如果两层的话就拼接起来
        else:
            hidden_cat=hidden[-1]
        fc_output=self.fc(hidden_cat)
        return fc_output
        
    #将codes和labels转换为tensor
    def make_tensors(self,codes,labels):
        sequences_and_lengths=[self.code2list(code) for code in codes]
        #分别将字符列表和长度取出来
        code_sequences=[s1[0] for s1 in sequences_and_lengths]
        seq_lengths=torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
        labels=labels.long()
        
        #padding：填充0使它们的长度相等
        #初始化一个全零的张量，然后把对应的字符复制过去
        seq_tensor=torch.zeros(len(code_sequences),seq_lengths.max()).long()
        for idx,(seq,seq_len) in enumerate(zip(code_sequences,seq_lengths),0):
            seq_tensor[idx,:seq_len]=torch.LongTensor(seq)
            
        #按照序列长度来排序
        seq_lengths,perm_idx=seq_lengths.sort(dim=0,descending=True)
        seq_tensor=seq_tensor[perm_idx]
        labels=labels[perm_idx]
        
        return self.create_tensor(seq_tensor),\
               self.create_tensor(seq_lengths),\
               self.create_tensor(labels)
    
    #将代码转化为字符串列表
    def code2list(self,code):
        arr=[ord(c) for c in code]
        return arr,len(arr)
    
     #判断是否放到显卡上
    def create_tensor(self,tensor):
        if USE_GPU:
            device=torch.device('cuda:0')
            tensor=tensor.to(device)
        return tensor 


            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
          

 
        