import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn#神经网络模块
import torch#用来创建张量、定义数据类型、做数学运算



#train_data = pd.read_csv('train.csv')
#双斜杠\\或r""编写硬编码模式
train_data = pd.read_csv(r"F:\Deep learning\house-prices-advanced-regression-techniques\train.csv")
test_data = pd.read_csv(r"F:\Deep learning\house-prices-advanced-regression-techniques\test.csv")
print(train_data.shape)
print(test_data.shape)
all_features=pd.concat([train_data.iloc[:,1:-1],test_data.iloc[:,1:]])
print(all_features.shape)

numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features]=all_features[numeric_features].apply(lambda x:((x-x.mean())/(x.std())))
all_features[numeric_features]=all_features[numeric_features].fillna(0)

#处理字符串
all_features=pd.get_dummies(all_features,dummy_na=True)
all_features.shape

n_train=train_data.shape[0]#拿到训练集样本数量，取表格的行数=样本个数
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)#把训练集的特征转换成张量，all_features[:n_train]表示取前n_train行，即训练集的特征，.values把DataFrame转换成numpy数组，dtype=torch.float32表示数据类型为32位浮点数
test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
print(train_features.shape)
print(test_features.shape)

loss=nn.MSELoss()
in_features=train_features.shape[1]#拿到的是特征数量
def get_net():
    net=nn.Sequential(
        nn.Linear(in_features,1)#nn.Linear(331,1)331个特征维度输入，最终只输出一个预测房价值
        )
    return net

#预测模型评价函数
def log_rmse(net,features,labels):#锅、菜、标准菜
    clipped_preds=torch.clamp(net(features), 1, float('inf'))#把预测值限制在1到正无穷之间，防止对数函数输入为负数或
                                                              #\torch.clamp(输入张量，最小值，最大值)
                                                              #float('inf')表示正无穷
    rmse= torch.sqrt(loss(clipped_preds, labels))
    return rmse.item()
def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]#存放误差的空表
    train_iter=load_array((train_features,train_labels),batch_size)#把数据切成一小批一小批的，方便喂给模型
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    for epochs in range(num_epochs):
        for x,y in train_iter:#逐批喂数据，每次喂一批数据，更新一次参数，是以batch_size为单位的迭代
            optimizer.zero_grad()#每次学习前，把上次的 “修改方向” 清零，不然会叠加混乱。
            l=loss(net(x),y)
            l.backward()#误差反向传播 → 告诉 w 和 b 该怎么改！
            optimizer.step()#更新模型权重
        train_ls.append(log_rmse(net, train_features, train_labels))#把这一轮的误差存起来
        if test_labels is not None:#如果有测试集，记录测试误差
            test_ls.append(log_rmse(net, test_features, test_labels))#返回测试误差
    return train_ls,test_ls     


#k折验证是用来在模型数据及比较小的时候，多次划分测试集和训练集，得到多种测试误差，求平均值，来评估模型的泛化能力，
# 这比单次划分测试集和训练集更可靠。可用在选择模型方面
def get_k_fold_data(k,i,X,y):#x是所有输入特征，y是所有标签
    assert k>1
    fold_size=X.shape[0]//k#每折的大小,样本数量/k
    X_train,y_train=None,None
    for j in range(k):
        idx= slice(j*fold_size,(j+1)*fold_size)#idx是一个切片范围
        X_part,y_part=X[idx,:],y[idx]#x是【取idx行，取所有列】，y是【取idx行】
        if j==i:#如果j等于i，说明这是第i折，作为验证集
            X_valid,y_valid=X_part,y_part
        elif X_train is None:#如果X_train还没有被定义，说明这是第一折，直接赋值
            X_train,y_train=X_part,y_part     
        else:#否则，这一折是训练集的一部分，把它和之前的训练集合并起来
            X_train=torch.cat((X_train,X_part),dim=0)#torch.cat()函数用于连接两个张量，dim=0表示按行连接
            y_train=torch.cat((y_train,y_part),dim=0)           
    return X_train,y_train,X_valid,y_valid    


#训练k次，并返回训练和验证的平均误差
def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decacy,batch_size):
   train_l_sim,valid_l_sim=0,0
   for i in range(k):
      data=get_k_fold_data(k,i,X_train,y_train)
      net=get_net()#每次训练都要重新实例化模型，参数会被
      train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decacy,batch_size)
      train_l_sim+=train_ls[-1]#train_ls中有epochs轮，每一轮的误差，把最后一轮train_ls[-1]的误差加起来，train_ls[-1]是一折的最后一轮误差，共有k折
      valid_l_sim+=valid_ls[-1]
      if i==0:#只在第一折画图，其他折的图都一样
         plt.plot(list(range(1,num_epochs+1)),train_ls)
         plt.plot(list(range(1,num_epochs+1)),valid_ls)#python的range[)左闭右开，range(1,num_epochs+1)表示从1到num_epochs的整数
                                                                  #xlim=[1,num_epochs]，x轴范围从1到num_epochs
         plt.xlabel('epoch')
         plt.ylabel('rmse')
         plt.xlim([1,num_epochs])
         plt.legend(['train','valid'])
         plt.show()
      print(f'折{i+1},训练误差{float(train_ls[-1]):f},验证误差{float(valid_ls[-1]):f}')
   return train_l_sim/k,valid_l_sim/k


k,num_epochs,lr,weight_decacy,batch_size=5,100,5,0,64#k是折数，num_epochs是训练轮数，lr是学习率，weight_decacy是权重衰减，batch_size是批量大小,都是超参数，手动调整
train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decacy,batch_size)#得到的是训练k次，并返回训练和验证的平均误差
print(f'{k}-折验证:平均训练误差{float(train_l):f},平均验证误差{float(valid_l):f}')#f' '，f-string 的标志，有了这个 f，大括号 {} 里的内容会被当成变量计算
                                                                                #float(train_l)把训练误差转成小数
                                                                                #:f → 按浮点数格式输出（默认保留 6 位小数）
def train_and_pred(train_features,train_labels,test_features,num_epochs,lr,weight_decacy,batch_size):
    net=get_net()
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decacy,batch_size)
    plt.plot(list(range(1,num_epochs+1)),train_ls,xlabel='epoch',ylabel='rmse',
             xlim=[1,num_epochs],legend=['train'])
    print(f'训练误差{float(train_ls[-1]):f}')
    preds=net(test_features).detach().numpy()#模型输出的张量 = 带计算图 + 带梯度，.detach()把预测结果从计算图中分离出来，转成numpy数组，
    
    test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])# 作用只有一个：把 [[a],[b],[c]] → [[a,b,c]]从3行1列变成1行3列，再取[0],即[a,b,c],[[]]必是二维表示
                                                            #pd.Series把一维数组 → 变成 pandas 专用的一列数据
    submission=pd.concat(test_data['Id'],test_data['SalePrice'],axis=1)#把id和预测的房价拼接成一个表格，axis=1表示按列拼接
    submission.to_csv('submission.csv',index=False)#把表格保存成csv文件，index=False表示不保存行索引
    
    train_and_pred(train_features,train_labels,test_features,num_epochs,lr,weight_decacy,batch_size)