import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import copy
import torch
import math
import networkx as nx
import torch_geometric
import torch.nn.functional as F
import time

from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from dataset1 import Mydataset
from torch.utils.data import DataLoader
from earlystop import EarlyStopping
from torch_cluster import knn_graph
from torch_geometric.utils import degree,to_undirected,to_networkx
from torch_geometric.nn import GCNConv,BatchNorm
from scipy import special


start = time.time()

## 考虑了阈值V的不确定性，theta的不确定性和扩散系数的不确定性，方差没法加入，只能用期望的形式，效果100
### FD001

seed=868 #随机种子
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.


early_stopping = EarlyStopping(20,verbose=True) #早停模块，如果test的指标在20次内不下降，取最好的一次，直接停止

device = torch.device("cuda")

# 读取数据
data = np.loadtxt("D:/中间估计器/2023年度工作/数模联动/RUL/CMAPSSDataNASA数据集六/train_FD001.txt", dtype=float)
data = data.astype(np.float32)
# 提出需要的传感器(数据集里面有部分数据不变，直接去掉)
data = data[:, [0, 1, 6, 7, 8, 11, 15, 16, 19, 21, 24, 25]]

# 计算真实RUL(开始数据的第二列是从1到最后寿命，这里做了一次颠倒)
for i in range(1, 101):
    data[np.where(data[:, 0] == i), 1] = np.max(data[np.where(data[:, 0] == i), 1]) - \
                                                       data[np.where(data[:, 0] == i), 1];
#上面得到的寿命是-1的，这里加上1
data[:, 1] = data[:, 1] + 1

data_num=np.empty((0),dtype="int")
data_list = []
## 去噪+最大最小归一化+构成list ##
for i in range(1, 101):
    jj = np.where(data[:, 0] == i)
    jj = np.array(jj)
    for j in range(2, 12):
        data[jj, j] = savgol_filter(data[jj, j], 9, 3)
        # data_choice[jj,j]=(data_choice[jj,j]-np.min(data_choice[jj,j]))/(np.max(data_choice[jj,j])-np.min(data_choice[jj,j]))
    data[jj, 2:12] = MinMaxScaler().fit_transform(np.squeeze(data[jj, 2:12]))  # 归一化
    data_num=np.concatenate([data_num,np.expand_dims(np.array(jj.shape[1],dtype="int"),axis=0)],0)
    data_list.append(np.squeeze(data[jj, 0:12]))
    


## KNN建图部分 ##
edge_index = knn_graph(torch.tensor(data[:,2:12]).T, k=5, loop=False, flow="target_to_source") #取10个传感器的数据，通过KNN建图,和邻近3个节点之间有边
edge_index = to_undirected(edge_index) #将有向图转换为无向图
edge_index=edge_index.to(device)
# print(edge_index)

data=torch.unsqueeze(torch.tensor(data[:,2:12]),2).to(device)


##图可视化（平时可不用
# edge_index=edge_index.T.tolist()
# G = nx.Graph()
# G.add_edges_from(edge_index)
# nx.draw(G,with_labels=True)
# plt.show()

## 100个设备，从每个的前50个中抽取300组，然后将100个拼成一组，每一组里面有所有的设备，每个设备的数据是从前50个中随机抽取的
N=300
data_choose=[]
data_mowei=[]
data_end=np.empty((0,12),dtype="float32")
for i in range(100):
    suoyin=np.random.choice(range(1, 51), N)
    data_choose.append(data_list[i][suoyin,:])
    data_mowei.append(np.expand_dims(data_list[i][-1,2:12],0))
    
data_mowei = np.array(data_mowei).reshape(-1, 10)
data_mowei=torch.unsqueeze(torch.tensor(data_mowei),2).to(device)    
data_mowei=data_mowei[0:60]
    
data_ready=[]
for i in range(N):
    data1=np.empty((0,12),dtype="float32")
    for j in range(60):
        data1=np.vstack((data1,data_choose[j][i,:]))
    data_ready.append(data1)
data_ready_train = np.array(data_ready).reshape(-1, 12)

## 构建训练数据集
data_ready_train = Mydataset(data_ready_train)
train_size=60
train_data = DataLoader(data_ready_train, batch_size=train_size)


# 创建网络模型
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1=GCNConv(1, 4)
        # self.bn1=BatchNorm(10)
        self.gcn2=GCNConv(4, 4)
        # self.bn2=BatchNorm(10)
        # self.fc=
        self.gcn3=GCNConv(4, 4)
        self.line=nn.Sequential(
            nn.Flatten(),
            nn.Linear(40, 24),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(24, 1),
        )
        self.miu_w = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.sigma_w=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.gamma=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.miu_w.size(0))
        self.miu_w.data.uniform_(1.5, 2)  # 随机化参数
        self.sigma_w.data.uniform_(0, stdv)  # 随机化参数
        self.gamma.data.zero_() # 随机化参数

    def forward(self, x1, edge_index):
        x1 = self.gcn1(x1, edge_index)
        x1 = F.relu(x1)
        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(x2)
        x1 = self.gcn3(x2, edge_index)+x1
        x1 = F.relu(x1)
        x1 = self.line(x1)
        return x1
    
gcn = GCN()
gcn=gcn.to(device)

# 损失函数
loss_fn = nn.MSELoss()
loss_fn=loss_fn.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(gcn.parameters(),lr=learning_rate,weight_decay=0.01)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 80

loss_train =[]
loss_test  =[]
accuracy_test=[]

# cons = ({'type': 'ineq', 'fun': lambda x: x[0:100]},
#         {'type': 'ineq', 'fun': lambda x: x[100:200]})
# x0=np.random.rand(200).astype(np.float32)
# temp=torch.ones([100,1]).to(device)

# # x有200个带求解变量
# def func(x,label_train):
#     L=0
#     for c in range(100):
#         log_det_s= (label_train[c].int().item()-2)*np.log(x[c])+np.log(x[c]+(label_train[c].int().item()-1)*x[c+100])
#         jiaocha=(1/x[c])-(x[c+100]/(x[c]**2+(label_train[c].int().item()-1)*x[c]*x[c+100]))
#         L=L+0.5*log_det_s+0.5*jiaocha
    # for i in range(HH):
        # iii=ii+data_num[i]
        # Q_inv=np.diag(2*np.ones(label_train[i].int().item()-1)) \
        #       +np.diag(-1*np.ones(label_train[i].int().item()-2),1) \
        #       +np.diag(-1*np.ones(label_train[i].int().item()-2),-1)
        # Q_inv[-1,-1]=1
        # Q_inv=np.float32(Q_inv)
        # Q=np.linalg.inv(Q_inv)

        # T=np.expand_dims(np.array(np.arange(1,label_train[i].item(),1),dtype=np.float32),axis=1)
        # O=np.expand_dims(np.array(np.zeros(label_train[i].int().item()-1),dtype=np.float32),axis=1)
        # O[-1]=1
    
        # log_det_s= (label_train[i].int().item()-1)*np.log(x[i])+(label_train[i].int().item()-2)*np.log(x[i])+np.log((label_train[i].int().item()-1)*x[i+HH])
        # S_inv= ((1/(x[i]))*Q_inv)-(x[i+1]/(x[i]**2+(label_train[i].int().item()-1)*(x[i])*(x[i+HH])))*O*O.T
        # miu_theta1=miu_theta[i].item()
        
        # # (T.T@S_inv@(data1[iii-label_train[i].int().item()+1:iii]-data1[iii-label_train[i].int().item()]*np.ones((label_train[i].int().item()-1,1),dtype="float32")))/(T.T@S_inv@T)
        # xx=(data1[iii-label_train[i].int().item()+1:iii]-data1[iii-label_train[i].int().item()]*np.ones((label_train[i].int().item()-1,1),dtype="float32")-miu_theta1*T)
        # aa=xx.T@S_inv@xx
        # ii=iii
        # L=L+0.5*log_det_s+0.5*aa[0]
        
        # log_det_s= (label_train[i].int().item()-2)*np.log(x[i])+np.log(x[i]+(label_train[i].int().item()-1)*x[i+HH])
        # jiaocha=(1/x[i])-(x[i+HH]/(x[i]**2+(label_train[i].int().item()-1)*x[i]*x[i+HH]))
        
        # S_inv= ((1/(x[i]))*Q_inv)-(x[i+1]/(x[i]**2+(label_train[i].int().item()-1)*(x[i])*(x[i+HH])))*O*O.T
        # miu_theta1=miu_theta[i].item()
        
        # # (T.T@S_inv@(data1[iii-label_train[i].int().item()+1:iii]-data1[iii-label_train[i].int().item()]*np.ones((label_train[i].int().item()-1,1),dtype="float32")))/(T.T@S_inv@T)
        # xx=(data1[iii-label_train[i].int().item()+1:iii]-data1[iii-label_train[i].int().item()]*np.ones((label_train[i].int().item()-1,1),dtype="float32")-miu_theta1*T)
        # aa=xx.T@S_inv@xx
        # ii=iii
        # L=L+0.5*log_det_s+0.5*jiaocha
    # return L


# x0=[0.01,0.00001]
# data2=data1.cpu().detach().numpy()
# cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
#         {'type': 'ineq', 'fun': lambda x: -x[0]+1},
#         {'type': 'ineq', 'fun': lambda x: x[1]-10e-5},
#         {'type': 'ineq', 'fun': lambda x: -x[1]+1},)

# args = 1

# miu_theta=0.00938914
# def func(args):
#     a=args
#     jiao=data2[45:191]-data2[44:190]-miu_theta*np.ones((146,1))
#     # np.concatenate([np.zeros([1,1]),data2[44:190]],0)
#     L=lambda x: 0.5*np.log(2*np.pi)*147+0.5*145*np.log(x[0])+0.5*np.log(x[0]+146*x[1]) \
#         +0.5*((np.matmul(jiao.T, jiao).item()/x[0])-x[1]*(data2[191]-data2[45]-146*miu_theta)/(x[0]*x[0]+146*x[0]*x[1]))
#     return L
# res = minimize(func(args), x0, method='SLSQP',constraints=cons)

# res.success
# res.x

# var=(1.5218-data2[45])*res.x[0]*miu_theta/(2*res.x[1]**2)-(1.5218-data2[45])*res.x[0]/(2*res.x[1]*miu_theta)-147*147
# var=np.sqrt(var) 





# miu_theta=0.00938914
# def func(args):
#     a=args
#     jiao=data2[45:191]-data2[44:190]-miu_theta*np.ones((146,1))
#     # np.concatenate([np.zeros([1,1]),data2[44:190]],0)
#     L=lambda x: 0.5*np.log(2*np.pi)*147+0.5*145*np.log(x[0])+0.5*np.log(x[0]+146*x[1]) \
#         +0.5*((np.matmul(jiao.T, jiao).item()/x[0])-x[1]*(data2[191]-data2[45]-146*miu_theta)/(x[0]*x[0]+146*x[0]*x[1]))
#     return L
# res = minimize(func(args), x0, method='SLSQP',constraints=cons)

# res.success
# res.x

# var=(1.5218-data2[45])*res.x[0]*miu_theta/(2*res.x[1]**2)-(1.5218-data2[45])*res.x[0]/(2*res.x[1]*miu_theta)-147*147
# var=np.sqrt(var) 


x0=[0.01,0.0001]
cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: -x[0]+1},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: -x[1]+1},)


for i in range(epoch):
    train_zl_list=[]
    print("-------第 {} 轮训练开始-------".format(i+1))
    
    # 训练步骤开始
    gcn.train()
    total_train_loss=0
    total_train_accuracy=0
    
    for train in train_data:
        data_train_qishi, label_train = train
        data_train_qishi=data_train_qishi.to(device)
        label_train=label_train.to(device)

        data_train_qishi = gcn(data_train_qishi, edge_index)
        data_train_mowei = gcn(data_mowei, edge_index)
        
        data1=gcn(data,edge_index)
        
        miu_theta=(data_train_mowei-data_train_qishi)/label_train
        

        
        # for c in range(100):
        # res = minimize(func, x0, args=(label_train), method="SLSQP" ,options={'maxiter': 3},constraints=cons)
        
        # sigma=torch.unsqueeze(torch.tensor(res.x[0:100]).float(), 1).to(device)
        # sigma_theta=torch.unsqueeze(torch.tensor(res.x[100:200]).float(),1).to(device)
        
        

        T_est=(gcn.miu_w-data_train_qishi)/miu_theta
        
        # if i>=25:
        #     ii=0
        #     data2=data1.cpu().detach().numpy()
        #     std=torch.zeros([0])
        #     for c in range(100):
        #         data3=data2[ii:ii+data_num[c]]
            
        #         jiao=data3[data_num[c]-label_train[c].int()+1:data_num[c]]\
        #             -data3[data_num[c]-label_train[c].int():data_num[c]-1]-miu_theta[c].item()*np.ones((label_train[c].int()-1,1))
                
        #         L=lambda x: 0.5*np.log(2*np.pi)*label_train[c].item()\
        #             +0.5*(label_train[c].item()-2)*np.log(x[0])+0.5*np.log(x[0]+(label_train[c].item()-1)*x[1]) \
        #                 +0.5*((np.matmul(jiao.T, jiao).item()/(x[0]+0.001))\
        #                       -x[1]*(data3[-1]-data3[data_num[c]-label_train[c].int()]-(label_train[c].item()-1)*miu_theta[c].item()) \
        #                           /(x[0]*x[0]+(label_train[c].item()-1)*x[0]*x[1]))
                            
        #         res = minimize(L, x0, method='SLSQP',constraints=cons)
            
        #         var=(gcn.miu_w.item()-data3[data_num[c]-label_train[c].int()])*res.x[0]*miu_theta[c].item()/(2*res.x[1]**2+0.00000001)\
        #             -(gcn.miu_w.item()-data3[data_num[c]-label_train[c].int()])*res.x[0]/(2*res.x[1]*miu_theta[c].item()+0.00000001)-T_est[c].item()*T_est[c].item()
        #         if np.isnan(var):
        #             var=np.array([0])
        #         if np.isinf(var):
        #             var=np.array([0])
        #         if var<0:
        #             var=np.array([0])
        #         std=torch.cat([std,torch.sqrt(torch.abs(torch.tensor([var])))],0) 
            
        #         # sigma=np.concatenate([sigma,res.x[0]],0)
        #         # sigma_theta=np.concatenate([sigma,res.x[1]],0)
        #         ii=ii+data_num[c]
        
        #     std=std.cuda()
        #     m,n=std.shape
        #     T_std=torch.std(label_train)*torch.ones([m,1]).cuda()
        
        # # V_est=((gcn.miu_w-data_train_qishi)*sigma*miu_theta)/(2*torch.pow(sigma_theta, 2)) \
        # #       -((gcn.miu_w-data_train_qishi)*sigma)/(2*sigma_theta*miu_theta)-torch.pow(T_est,2)
             
        # # -((gcn.miu_w-data_train_qishi)*sigma*miu_theta)/(2*torch.pow(sigma_theta, 2)) \
        # #       -torch.pow(T_est,2) \
        # #       +((gcn.miu_w-data_train_qishi)*sigma*(torch.pow(miu_theta,2)-sigma_theta))/(2*torch.pow(sigma_theta, 2)*miu_theta)
        # #       +(gcn.sigma_w+torch.pow((gcn.miu_w-data_train_qishi),2))/(sigma_theta) 
        # #       -(gcn.sigma_w+torch.pow((gcn.miu_w-data_train_qishi),2))/sigma_theta \
                
        # # V=torch.sqrt(torch.var(label_train))*temp
        
        ##计算准确度.
        error_train=label_train-T_est
        zero = torch.zeros_like(error_train)
        one = torch.ones_like(error_train)
        error_train = torch.where(error_train < -10, zero, error_train)
        error_train = torch.where(error_train > 13, zero, error_train)
        train_accuracy = torch.count_nonzero(error_train)
        total_train_accuracy = total_train_accuracy + train_accuracy.item()  
        
        loss = loss_fn(T_est, label_train)/train_size \
             +(1/(100*torch.maximum(torch.tensor(0).to(device),torch.mean(data_train_mowei-gcn.miu_w))+0.01)) 
        # if i>=25:
        #     loss = torch.sigmoid(gcn.gamma)*loss_fn(T_est, label_train)/train_size + (1-torch.sigmoid(gcn.gamma))*torch.log10(1+loss_fn(std,T_std))\
        #         +(1/(100*torch.maximum(torch.tensor(0).to(device),torch.mean(data_train_mowei-gcn.miu_w))+0.01)) 
        # # + 0.01*(1-torch.sigmoid(gcn.gamma))*loss_fn(V_est,V)/train_size
              
        total_train_loss = total_train_loss + loss.item()

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            print("训练次数：{}, Loss: {}, Gamma: {}".format(total_train_step, loss.item(),gcn.gamma.item()))

    
    total_train_accuracy = total_train_accuracy/180   
    # total_test_accuracy = total_test_accuracy*100/5000
    print("训练集准确率 {}%".format(total_train_accuracy))
    # print("测试集准确率 {}%".format(total_test_accuracy))

    if total_train_accuracy==100:
        end=time.time()
    # loss_train.append(total_train_loss)
    # loss_test.append(total_test_loss)
    # accuracy_test.append(total_test_accuracy)
    
    # # early_stopping(accuracy_test[i], gcn)
    # # #达到早停止条件时，early_stop会被置为True
    # # if early_stopping.early_stop:
    # #     print("Early stopping")
    # #     break #跳出迭代，结束训练
    
    if i % 2 == 0:
        test1=data_list[0][:,2:12]
        lll=test1.shape[0]
        onehot=np.ones((lll, 1))
    
        test1=torch.Tensor(test1).to(device)
        test1=torch.unsqueeze(test1, 2)
        xx2 = gcn(test1,edge_index)
        xx2=torch.Tensor.cpu(xx2).detach().numpy()
        threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
        onehot=threshold*onehot
    
        plt.figure(1)
        plt.plot(xx2)
        plt.plot(onehot)
        plt.show()


    
#     # # 测试步骤开始
#     # gcn.eval()
#     # total_test_loss = 0
#     # total_test_accuracy = 0
#     # with torch.no_grad():
#     #     for data_test in test_dataloader:
#     #         data_row_test, data_zuocha_test, label_test = data_test
            
#     #         data_row_test = data_row_test.to(device)
#     #         data_zuocha_test = data_zuocha_test.to(device)
#     #         label_test = label_test.to(device)
#     #         edge_index = edge_index.to(device)

#     #         fenzi,fenmu = gcn(data_row_test,data_zuocha_test,edge_index)
#     #         rul_est=(torch.mul(label_test,fenzi))/fenmu
            
#     #         ##计算准确度
#     #         error_test=label_test-rul_est
#     #         zero = torch.zeros_like(error_test)
#     #         one = torch.ones_like(error_test)
#     #         error_test = torch.where(error_test < -5, zero, error_test)
#     #         error_test = torch.where(error_test > 5, zero, error_test)
#     #         test_accuracy = torch.count_nonzero(error_test)
#     #         total_test_accuracy = total_test_accuracy + test_accuracy.item()  
            
            
#     #         loss = loss_fn(rul_est, label_test)/size
#     #         total_test_loss = total_test_loss + loss.item()
            

    
# torch.save(gcn.state_dict(),"gcn_FD001.pth") #将vgg16中模型参数保存为字典形式
# gcn.load_state_dict(torch.load("gcn_FD001.pth"))


#%%################ 单个发动机的曲线
test1=data_list[65][:,1:12]
lll=test1.shape[0]
onehot=np.ones((lll, 1))

ttt=torch.Tensor(test1[:,1:12]).unsqueeze(2).to(device)
xx2 = gcn(ttt,edge_index)

miumiu=(xx2[-1]-xx2[:-1])/torch.tensor(test1[:-1,0]).unsqueeze(1).to(device)
est=(gcn.miu_w-xx2[:-1])/miumiu

### 查看单个发动机寿命
plt.figure(1)
plt.subplot(421)
plt.grid(True)
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=0.55)
plt.plot(test1[:,0],'b',linewidth=2,label='Actual lifetime')
plt.plot(est.cpu().detach().numpy(),'r--',linewidth=2,label='Predicted lifetime')

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 10,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 8,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
# plt.xlabel('Test Engine #65',font1)
plt.ylabel('RUL',font1)
plt.title('Test Engine #65',font1)

# # 查看最后的复合性能指标
xx2=torch.Tensor.cpu(xx2).detach().numpy()
threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
onehot=threshold*onehot

# plt.subplot(422)
plt.grid(True)
plt.plot(xx2,'b',label='CHI')
plt.plot(onehot,'r',label='Threshold')
plt.legend(loc='upper left',edgecolor='black',prop=font2,ncol=2)
# plt.xlim(0,18)#X轴范围
# plt.ylim(-1,3)#显示y轴范围
plt.title('Test Engine #65',font1)
plt.show()

#############
test1=data_list[85][:,1:12]
lll=test1.shape[0]
onehot=np.ones((lll, 1))

ttt=torch.Tensor(test1[:,1:12]).unsqueeze(2).to(device)
xx2 = gcn(ttt,edge_index)

miumiu=(xx2[-1]-xx2[:-1])/torch.tensor(test1[:-1,0]).unsqueeze(1).to(device)
est=(gcn.miu_w-xx2[:-1])/miumiu

### 查看单个发动机寿命
plt.subplot(423)
plt.grid(True)
plt.plot(test1[:,0],'b',linewidth=2,label='Actual lifetime')
plt.plot(est.cpu().detach().numpy(),'r--',linewidth=2,label='Predicted lifetime')
font1 = {'family' : 'Arial','weight' : 'normal','size'   : 10,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 8,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
# plt.xlabel('Test Engine #85',font1)
plt.ylabel('RUL',font1)
plt.title('Test Engine #85',font1)

# # 查看最后的复合性能指标
xx2=torch.Tensor.cpu(xx2).detach().numpy()
threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
onehot=threshold*onehot

plt.subplot(424)
plt.grid(True)
plt.plot(xx2,'b',label='CHI')
plt.plot(onehot,'r',label='Threshold')
plt.legend(loc='upper right',edgecolor='black',prop=font2,ncol=2)
# plt.xlim(0,18)#X轴范围
plt.ylim(-1,3)#显示y轴范围
plt.title('Test Engine #85',font1)

##############
test1=data_list[90][:,1:12]
lll=test1.shape[0]
onehot=np.ones((lll, 1))

ttt=torch.Tensor(test1[:,1:12]).unsqueeze(2).to(device)
xx2 = gcn(ttt,edge_index)

miumiu=(xx2[-1]-xx2[:-1])/torch.tensor(test1[:-1,0]).unsqueeze(1).to(device)
est=(gcn.miu_w-xx2[:-1])/miumiu

### 查看单个发动机寿命
plt.subplot(425)
plt.grid(True)
plt.plot(test1[:,0],'b',linewidth=2,label='Actual lifetime')
plt.plot(est.cpu().detach().numpy(),'r--',linewidth=2,label='Predicted lifetime')
font1 = {'family' : 'Arial','weight' : 'normal','size'   : 10,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 8,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
# plt.xlabel('Test Engine #90',font1)
plt.ylabel('RUL',font1)
plt.title('Test Engine #90',font1)

# # 查看最后的复合性能指标
xx2=torch.Tensor.cpu(xx2).detach().numpy()
threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
onehot=threshold*onehot

# plt.subplot(426)
plt.grid(True)
plt.plot(xx2,'b',label='CHI')
plt.plot(onehot,'r',label='Threshold')
plt.legend(loc='upper left',edgecolor='black',prop=font2,ncol=2)
# plt.xlim(0,18)#X轴范围
# plt.ylim(-1,3)#显示y轴范围
plt.title('Test Engine #90',font1)
plt.show()

##############
test1=data_list[95][:,1:12]
lll=test1.shape[0]
onehot=np.ones((lll, 1))

ttt=torch.Tensor(test1[:,1:12]).unsqueeze(2).to(device)
xx2 = gcn(ttt,edge_index)

miumiu=(xx2[-1]-xx2[:-1])/torch.tensor(test1[:-1,0]).unsqueeze(1).to(device)
est=(gcn.miu_w-xx2[:-1])/miumiu

### 查看单个发动机寿命
plt.subplot(427)
plt.grid(True)
plt.plot(test1[:,0],'b',linewidth=2,label='Actual lifetime')
plt.plot(est.cpu().detach().numpy(),'r--',linewidth=2,label='Predicted lifetime')
font1 = {'family' : 'Arial','weight' : 'normal','size'   : 10,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 8,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
plt.xlabel('Cycle',font1)
plt.ylabel('RUL',font1)
plt.title('Test Engine #95',font1)

# # 查看最后的复合性能指标
xx2=torch.Tensor.cpu(xx2).detach().numpy()
threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
onehot=threshold*onehot

plt.subplot(428)
plt.grid(True)
plt.plot(xx2,'b',label='CHI')
plt.plot(onehot,'r',label='Threshold')
plt.show()
plt.legend(loc='upper right',edgecolor='black',prop=font2,ncol=2)
plt.xlabel('Cycle',font1)
# plt.xlim(0,18)#X轴范围
plt.ylim(-1,3)#显示y轴范围
plt.title('Test Engine #95',font1)




#%%# #测试用
# #100个设备从起始到末尾的寿命估计

data_zuihou=np.zeros([100 ,12],dtype="float32")  
data_qishi=np.zeros([100 ,12],dtype="float32")  


for i in range(100):
    data_qishi[i,:]=data_list[i][0,:]
    data_zuihou[i,:]=data_list[i][-1,:]


label1=torch.unsqueeze(torch.Tensor(data_qishi[:,1]), 1)
data_qishi=torch.unsqueeze(torch.Tensor(data_qishi[:,2:12]), 2)
data_zuihou=torch.unsqueeze(torch.Tensor(data_zuihou[:,2:12]), 2)



data_qishi=data_qishi.to(device)
data_zuihou=data_zuihou.to(device)
label1=label1.to(device)

# d1 = gcn(abab,edge_index)
# d2 = gcn(acac,edge_index)

theta_i=(gcn(data_zuihou,edge_index)-gcn(data_qishi,edge_index))/label1
# sigma_i=(train_zl_tensor-((gcn(data_zuocha_train,edge_index)**2)/label_train))/label_train


rul_ceshi_est=(gcn.miu_w-gcn(data_qishi,edge_index))/theta_i

# rul_ceshi_est=(torch.mul(label1,d1))/d2

rul_ceshi_est=torch.Tensor.cpu(rul_ceshi_est)
label1=torch.Tensor.cpu(label1)

rul_ceshi_est=rul_ceshi_est.detach().numpy()
label1=label1.detach().numpy()


aa=np.linspace(61, 100,40)

plt.subplot(121)
plt.grid(True)
plt.plot(aa,label1[60:100],'b',marker='o',markeredgecolor = 'b',markerfacecolor=(0, 0, 0, 0),linewidth=2,label='Actual lifetime')
plt.plot(aa,rul_ceshi_est[60:100],'r--',marker='s',markeredgecolor = 'r',markerfacecolor=(0, 0, 0, 0),linewidth=2,label='Predicted lifetime')

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 14,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 12,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
plt.xlabel('Engine ID',font1)
plt.ylabel('RUL',font1)
# plt.xlim(0,18)#X轴范围
plt.ylim(120,400)#显示y轴范围
# plt.show()


# ######
err=label1[60:100]-rul_ceshi_est[60:100]
plt.subplot(122)
plt.grid(True)
plt.plot(aa,err,'b',linewidth=2,label='Prediction error')

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 14,}
font2 = {'family' : 'Arial','weight' : 'normal','size'   : 12,}
plt.legend(loc='upper right',edgecolor='black',prop=font2)
plt.xlabel('Engine ID',font1)
# plt.ylabel('RUL',font1)
# plt.xlim(0,18)#X轴范围
plt.ylim(-2,8)#显示y轴范围
plt.show()


####RMSE
MSE=np.sum(err**2)/40
####Accuracy
jishu=np.zeros([40,1])
acc = np.where(err < -10, jishu, err)
acc = np.where(err >13, jishu, acc)
acc = np.count_nonzero(acc)

Accuracy= acc*100/40
####Score
scoz=err[np.where(err>0)]
scof=err[np.where(err<0)]
sz=np.sum(np.exp(scoz/13)-1)
sf=np.sum(np.exp(-scof/10)-1)
score=(sz+sf)
####R2
L_mean=np.mean(label1[60:100])
fenmu=np.sum((label1[60:100]-L_mean)**2)
fenzi=np.sum(err**2)
R2=1-fenzi/fenmu

print(MSE,Accuracy,score,R2)
