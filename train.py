# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 18:27:04 2021

@author: 10983
"""
import torch
import time
import math

class Train_Valid():
    def __init__(self,model,trainloader,validloader,start,trainset,validset):
        self.model=model
        self.trainloader=trainloader
        self.validloader=validloader
        self.start=start
        self.trainset=trainset
        self.validset=validset
        self.gamma=0.1
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.01)
        self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,self.gamma, last_epoch=-1)
        

    def train(self):
        acc_list=[]
        #循环100轮
        loss_list=[]
        #min_loss=100  #初始化一个最小损失值
        for epoch in range(1,31): 
            total_loss=0
            for i,(inputs,target) in enumerate(self.trainloader,1):
                self.optimizer.zero_grad()
                output=self.model.forward(inputs)
                loss=self.criterion(output,target)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
            
                total_loss+=loss.item()
                if i%500==0:
                    print(f'[{self.time_since(self.start)}] Epoch {epoch}',end='')
                    print(f'[{i * len(inputs)}/{len(self.trainset)}]',end='')
                    print(f'loss={total_loss/(i*len(inputs))}')
    
            '''
            print('train网络的参数：')
            i=0
            for parameters in self.model.parameters():
                 print(parameters)
                 i+=1
                 if i==4:
                     break
            '''
            
            
            '''
            #如果出现损失值的最新低值，则保存网络模型
            if (total_loss/len(trainset))<min_loss:
                print('**********************')
                min_loss=total_loss/len(trainset) #更新值
                torch.save(classifier.state_dict(),'C:/Users/10983/py入门/GRUClassifier/net_params.pkl')
            '''
            loss_list.append(total_loss/len(self.trainset))
            #训练一轮的结束进行一次验证，获取准确率
            acc=self.valid(epoch)
            acc_list.append(acc)
        return acc_list,loss_list
    
    def valid(self,epoch):
        correct,test_total_loss=0,0
        total=len(self.validset)
        print('evaluating trained model...')
        #model.load_state_dict(torch.load('C:/Users/10983/py入门/GRUClassifier/net_params.pkl'))
        
        '''
        print()
        print('test网络的参数：')
        i=0
        for parameters in self.model.parameters():
            print(parameters)
            i+=1
            if i==4:
                break
        '''
        
        with torch.no_grad():#已经设置为验证模式 不会再求梯度
            for i,(inputs,target) in enumerate(self.validloader,1):
                output=self.model.forward(inputs)
                test_loss=self.criterion(output,target)
                pred=output.max(dim=1,keepdim=True)[1]
                correct+=pred.eq(target.view_as(pred)).sum().item()
                
                test_total_loss+=test_loss.item()
                if i%100==0:
                    print(f'[{self.time_since(self.start)}] Epoch {epoch}',end='')
                    print(f'[{i * len(inputs)}/{len(self.validset)}]',end='')
                    print(f'test_loss={test_total_loss/(i*len(inputs))}')
               
        percent='%.2f' % (100 * correct /total)
        print(f'Test set: Accuracy {correct}/{total} {percent} %')
            
        return correct / total  
    
    def time_since(self,since):
        s=time.time()-since
        m=math.floor(s/60)
        s-=m*60
        return '%dm %ds'% (m,s)





















