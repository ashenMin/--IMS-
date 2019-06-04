#!/usr/bin/env python
# coding: utf-8

# In[1]:


#encoding=utf-8
import pandas as pd
import numpy as np
import os
import pylab
import matplotlib.pyplot as plt
from scipy import stats, fftpack
import sklearn
import sklearn.metrics
from numpy import square, sqrt, array, arange, mean, var, linspace
from scipy import stats, fftpack
from sklearn import preprocessing


# In[2]:


os.chdir('D:/UNIVERSITY_STUDY/PHM/2nd_term/NASA PHM/IMS/2nd_test/2nd_test/')
file_chdir=os.getcwd()
set1=[]
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        File=pd.read_csv(file,header=None,sep='\s+',engine='python')
        set1.append(File)


# In[3]:


set1[0]


# In[4]:


set_size = 984 #数据集大小


# In[5]:


set2=[]
plt.figure(figsize=(12,5))  #利用快速傅立叶变换将2号轴承时域转化为频域
x2 = np.linspace(0, 1, 12000)
i=0
for i in range(set_size):
    y2 = set1[i][:][1]
    yy = fftpack.fft(y2)
    yf1=abs(fftpack.fft(y2))/len(x2)           #归一化处理
    xf1 = np.arange(len(y2))   # 频率
    set2.append(yf1)
plt.subplot(223)
plt.plot(xf1,yf1,'r')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
plt.show()


# In[6]:


set2


# In[7]:


import math
Rs0=1
Rs1=[]
Rs2=[]
Rs3=[]
Rs4=[]
Rs5=[]

def RS_value(data_set, freq1, freq2):
    Rs_val=[]
    fengzi=0
    qq1=0
    qq2=0
    fengmu1=0
    fengmu2=0
    fengmu=0
    for j in range(set_size):
        f0_avg=np.mean(data_set[0])
        f1_avg=np.mean(data_set[j])
        for i in range(freq1,freq2):
            qq1=data_set[0][i]-f0_avg
            qq2=data_set[j][i]-f1_avg
            fengzi=np.abs(fengzi+qq1*qq2)
            fengmu1=fengmu1+qq1*qq1
            fengmu2=fengmu2+qq2*qq2
        fengmu=fengmu1*fengmu2
        Rs_val.append(fengzi/math.sqrt(fengmu1*fengmu2))
    return Rs_val 

Rs1=RS_value(set2, 0, 2560)
Rs2=RS_value(set2, 2560, 5120)
Rs3=RS_value(set2, 5120, 7680)
Rs4=RS_value(set2, 7680, 10240)
Rs5=RS_value(set2, 0, 10240)


# In[8]:


#提取2号轴承均方根RMSE---------------1
i = 0
rmse = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    rmse.append(sqrt(sklearn.metrics.mean_squared_error(x,y)))
plt.plot(xs,rmse,'r')
plt.xlabel(u"Time domain-RMSE") #X轴标签
plt.ylabel("RMSE/g") #Y轴标签
plt.show()

#提取2号轴承方差VAR-------------2
i = 0
var1 = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    var1.append(var(y))
plt.plot(xs,var1,'r')
plt.xlabel(u"Time domain-VAR") #X轴标签
plt.ylabel("VAR") #Y轴标签
plt.show()

#提取2号轴承绝对均值-----------------------------3
i = 0
j = 0
mean1 = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    y1 = array([y]).T
    for j in range(20480):
        y1[j]=abs(y1[j])
    mean1.append(mean(y1))
plt.plot(xs,mean1,'r')
plt.xlabel(u"Time domain-mean") #X轴标签
plt.ylabel("mean/g") #Y轴标签
plt.show()

#提取2号轴承峰度-------------------------4
i = 0
j = 0
ku_2 = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    ku_2.append(stats.kurtosis(y))
plt.plot(xs,ku_2,'r')
plt.xlabel(u"Time domain-ku") #X轴标签
plt.ylabel("ku") #Y轴标签
plt.show()

#提取2号轴承偏度----------5
i = 0
j = 0
skew = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    s = pd.Series(y)
    skew.append(s.skew())
plt.plot(xs,skew,'r')
plt.xlabel(u"Time domain-skew") #X轴标签
plt.ylabel("skew") #Y轴标签
plt.show()

#提取2号轴承峰峰值----------------6
i = 0
j = 0
ptp_2 = []
x = range(20480)
xs = range(set_size)
for i in range(set_size):
    y = set1[i][:][1]
    ptp_2.append(max(y) - min(y))
plt.plot(xs,ptp_2,'r')
plt.xlabel(u"Time domain-ptp") #X轴标签
plt.ylabel("peak-to-peak value") #Y轴标签
plt.show()


# In[9]:


#装向量
set3 = [[0] * 6 for i in range(set_size)]
rmse=preprocessing.scale(rmse)
var1=preprocessing.scale(var1)
mean1=preprocessing.scale(mean1)
ku_2=preprocessing.scale(ku_2)
skew=preprocessing.scale(skew)
ptp_2=preprocessing.scale(ptp_2)
for i in range(set_size):
    set3[i][0]=rmse[i]
    set3[i][1]=var1[i]
    set3[i][2]=mean1[i]
    set3[i][3]=ku_2[i]
    set3[i][4]=skew[i]
    set3[i][5]=ptp_2[i]
print(set3)


# In[10]:


#把向量转化为Rs向量
Rst1=RS_value(set3, 0, 6)

plt.plot(xs,Rst1,'r')
plt.xlabel(u"Time domain-RSt") #X轴标签
plt.ylabel("Rst value") #Y轴标签
plt.show()
print(Rst1)


# In[11]:


#开始尝试进行小波包分解

import pywt
import pywt.data

mode = pywt.Modes.smooth
set4=[]

def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    for i in range(3):
        (a, d) = pywt.dwt(a, w, mode)#进行3阶离散小波变换
        ca.append(a)

    rec_a = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构
    set4.append(rec_a[2])
for i in range(set_size):
    plot_signal_decomp(set1[i][1], 'sym5', "DWT: Ecg sample - Symmlets5")
print(set4)


# In[12]:


print(len(set4[0]))


# In[13]:


#计算能量比
#分为8段：0-2560；2560-5120；5120-7680；7680-10240
#10240-12800;12800-15360;15360-17920;17920-20480
Et=[]
for i in range(set_size):
    E1=0
    for j in range(2560):
        E1=E1+np.abs(set4[i][j])*np.abs(set4[i][j])

    E2=0
    for j in range(2560,5120):
        E2=E2+np.abs(set4[i][j])*np.abs(set4[i][j])

    E3=0
    for j in range(5120,7680):
        E3=E3+np.abs(set4[i][j])*np.abs(set4[i][j])

    E4=0
    for j in range(7680,10240):
        E4=E4+np.abs(set4[i][j])*np.abs(set4[i][j])

    E5=0
    for j in range(10240,12800):
        E5=E5+np.abs(set4[i][j])*np.abs(set4[i][j])

    E6=0
    for j in range(12800,15360):
        E6=E6+np.abs(set4[i][j])*np.abs(set4[i][j])

    E7=0
    for j in range(15360,17920):
        E7=E7+np.abs(set4[i][j])*np.abs(set4[i][j])

    E8=0
    for j in range(17920,20480):
        E8=E8+np.abs(set4[i][j])*np.abs(set4[i][j])
    Ez=E1+E2+E3+E4+E5+E6+E7+E8
    Et.append([E1/Ez,E2/Ez,E3/Ez,E4/Ez,E5/Ez,E6/Ez,E7/Ez,E8/Ez])

print(Et)


# In[14]:


#构建特征集
feature= [[0] * 14 for i in range(set_size)]
for i in range(set_size):
    feature[i][0]=Rs1[i]
    feature[i][1]=Rs2[i]
    feature[i][2]=Rs3[i]
    feature[i][3]=Rs4[i]
    feature[i][4]=Rs5[i]
    feature[i][5]=Rst1[i]
    feature[i][6]=Et[i][0]
    feature[i][7]=Et[i][1]
    feature[i][8]=Et[i][2]
    feature[i][9]=Et[i][3]
    feature[i][10]=Et[i][4]
    feature[i][11]=Et[i][5]
    feature[i][12]=Et[i][6]
    feature[i][13]=Et[i][7]


# In[15]:


feature[0]


# In[16]:


#corr计算
Zcorr=[]
for j in range(14):
    fengmu=0
    fengzi=0
    qq1=0
    qq2=0
    corr=0
    ss1=[]
    #计算第一个特征的corr
    for i in range(set_size):
        ss1.append(feature[i][j])
    Fb=np.mean(ss1)
    Lb=(set_size-1)/2
    for i in range(set_size):
        fengzi=fengzi+np.abs((ss1[i]-Fb)*(i-Lb))
        qq1=qq1+(ss1[i]-Fb)*(ss1[i]-Fb)
        qq2=qq2+(i-Lb)*(i-Lb)
    fengmu=math.sqrt(qq1*qq2)
    corr=fengzi/fengmu
    Zcorr.append(corr)
print(Zcorr)


# In[17]:


#计算Mon

ZMon=[]
for j in range(14):
    ss2=[]
    Zeng=0
    Fu=0
    #计算第一个特征的Mon
    for i in range(set_size):
        ss2.append(feature[i][j])
    for i in range(set_size-1):
        if(ss2[i+1]-ss2[i]>0):
            Zeng=Zeng+1
        if(ss2[i+1]-ss2[i]<0):
            Fu=Fu+1
    Mon=np.abs((Zeng-Fu)/(set_size-1))
    ZMon.append(Mon)
print(ZMon)


# In[18]:


#计算敏感特性的标准Cri=（Corr+Mon）/2
ZCri=[]
for i in range(14):
    ZCri.append((Zcorr[i]+ZMon[i])/2)
print(ZCri)


# In[19]:


#故障标签
data_lable = []
for i in range(set_size):
    data_lable.append((1/(set_size-1))*i)
data_lable


# In[22]:


#最终特征集,基于敏感指标的大小排序，丢弃了Rs3和Rs5这两个特征
feature_fin = [[0] * 13 for i in range(set_size)]
for i in range(set_size):
    feature_fin[i][0] = feature[i][0]
    feature_fin[i][1] = feature[i][1]
    feature_fin[i][2] = feature[i][3]
    feature_fin[i][3] = feature[i][5]
    feature_fin[i][4] = feature[i][6]
    feature_fin[i][5] = feature[i][7]
    feature_fin[i][6] = feature[i][8]
    feature_fin[i][7] = feature[i][9]
    feature_fin[i][8] = feature[i][10]
    feature_fin[i][9] = feature[i][11]
    feature_fin[i][10] = feature[i][12]
    feature_fin[i][11] = feature[i][13]
    feature_fin[i][12] = data_lable[i]


# In[50]:


len(feature_fin)


# In[38]:


#将时间序列数据转化为有监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#转换数据集
values = np.array(feature_fin)
re_feature = series_to_supervised(values)
# drop columns we don't want to predict
re_feature.drop(re_feature.columns[[13,14,15,16,17,18,19,20,21,22,23,24]], axis=1, inplace=True)


# In[51]:


re_feature


# 开始建立预测模型

# In[69]:


#取偶数点和奇数点分别作为训练集和测试集
values = re_feature.values
train_data = []#训练集
valid_data = []#验证集
test_data = []#测试集
for i in range(set_size-1):
    if i%3 == 0:
        valid_data.append(values[i])
    elif i%3 == 1:
        test_data.append(values[i])
    else:
        train_data.append(values[i])
train_data = np.array(train_data)
valid_data = np.array(valid_data)
test_data = np.array(test_data)
train_x, train_y = train_data[:, :-1], train_data[:, -1]
valid_x, valid_y = valid_data[:, :-1], valid_data[:, -1]
test_x, test_y = test_data[:, :-1], test_data[:, -1]
#将训练集和测试集转换成符合LSTM要求的数据格式,即 [样本，时间步，特征]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
valid_x = valid_x.reshape((valid_x.shape[0], 1, valid_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape, valid_x.shape, valid_y.shape)


# In[70]:


#通过keras框架建立LSTM网络模型并训练
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(50, activation='relu',input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam') 
# fit network
LSTM = model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(valid_x, valid_y), verbose=2, shuffle=False)
# plot history
plt.plot(LSTM.history['loss'], label='train')
plt.plot(LSTM.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[71]:


#带入测试集进行预测
xl=range(328)
plt.figure(figsize=(24,8))
test_predict = model.predict(test_x)
plt.plot(xl,test_y,'r')
plt.plot(xl,test_predict,'b')
plt.show()


# In[ ]:




