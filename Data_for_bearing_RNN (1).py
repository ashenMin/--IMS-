#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        File=pd.read_csv(file,sep='\s+',header=None,engine='python')
        set1.append(File)
print(set1)


# In[3]:


set2=[]
plt.figure(figsize=(12,5))  #利用快速傅立叶变换将2号轴承时域转化为频域
x2 = np.linspace(0, 1, 12000)
i=0
for i in range(984):
    y2 = set1[i][:][1]
    yy = fftpack.fft(y2)
    yf1=abs(fftpack.fft(y2))/len(x2)           #归一化处理
    xf1 = np.arange(len(y2))   # 频率
    set2.append(yf1)
plt.subplot(223)
plt.plot(xf1,yf1,'r')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
plt.show()


# In[4]:


print(set2)


# In[5]:


for i in range(0,10):
    print(i)


# In[6]:


import math
Rs0=1
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
Rs1=[]
Rs2=[]
Rs3=[]
Rs4=[]
Rs5=[]
for j in range(984):
    f0_avg=np.mean(set2[0])
    f1_avg=np.mean(set2[j])
    for i in range(2560):
        qq1=set2[0][i]-f0_avg
        qq2=set2[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    Rs1.append(fengzi/math.sqrt(fengmu1*fengmu2))
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
for j in range(984):
    f0_avg=np.mean(set2[0])
    f1_avg=np.mean(set2[j])
    for i in range(2560,5120):
        qq1=set2[0][i]-f0_avg
        qq2=set2[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    Rs2.append(fengzi/math.sqrt(fengmu1*fengmu2))
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
for j in range(984):
    f0_avg=np.mean(set2[0])
    f1_avg=np.mean(set2[j])
    for i in range(5120,7680):
        qq1=set2[0][i]-f0_avg
        qq2=set2[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    Rs3.append(fengzi/math.sqrt(fengmu1*fengmu2))
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
for j in range(984):
    f0_avg=np.mean(set2[0])
    f1_avg=np.mean(set2[j])
    for i in range(7680,10240):
        qq1=set2[0][i]-f0_avg
        qq2=set2[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    Rs4.append(fengzi/math.sqrt(fengmu1*fengmu2))
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
for j in range(984):
    f0_avg=np.mean(set2[0])
    f1_avg=np.mean(set2[j])
    for i in range(10240):
        qq1=set2[0][i]-f0_avg
        qq2=set2[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    Rs5.append(fengzi/math.sqrt(fengmu1*fengmu2))

    


    
    
    
    
print(Rs1)
print(Rs2)
print(Rs3)
print(Rs4)
print(Rs5)


# In[7]:


#提取2号轴承均方根误差RMSE---------------1
i = 0
rmse_1 = []
x = range(20480)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    rmse_1.append(sqrt(sklearn.metrics.mean_squared_error(x,y)))
plt.plot(xs,rmse_1,'r')
plt.xlabel(u"Time domain-RMSE") #X轴标签
plt.ylabel("RMSE/g") #Y轴标签
plt.show()

#提取2号轴承方差VAR-------------2
i = 0
var1 = []
x = range(20480)
xs = range(984)
for i in range(984):
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
xs = range(984)
for i in range(984):
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
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    ku_2.append(stats.kurtosis(y))
plt.plot(xs,ku_2,'r')
plt.xlabel(u"Time domain-ku") #X轴标签
plt.ylabel("ku") #Y轴标签
plt.show()

#提取2号轴承方根幅值-----------5
i = 0
j = 0
RMSA_2 = []
x = range(20480)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    y1 = array([y]).T
    for j in range(20480):
        y1[j]=abs(y1[j])
    RMSA_2.append(square(sum(sqrt(y1) / 20480)))
plt.plot(xs,RMSA_2,'r')
plt.xlabel(u"Time domain-RMSA") #X轴标签
plt.ylabel("RMSA") #Y轴标签
plt.show()

#提取2号轴承峰峰值----------------6
i = 0
j = 0
ptp_2 = []
x = range(20480)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    ptp_2.append(max(y) - min(y))
plt.plot(xs,ptp_2,'r')
plt.xlabel(u"Time domain-ptp") #X轴标签
plt.ylabel("peak-to-peak value") #Y轴标签
plt.show()


# In[8]:


from sklearn import preprocessing
#装向量
set3 = [[0] * 6 for i in range(984)]
rmse_1=preprocessing.scale(rmse_1)
var1=preprocessing.scale(var1)
mean1=preprocessing.scale(mean1)
ku_2=preprocessing.scale(ku_2)
RMSA_2=preprocessing.scale(RMSA_2)
ptp_2=preprocessing.scale(ptp_2)
for i in range(984):
    set3[i][0]=rmse_1[i]
    set3[i][1]=var1[i]
    set3[i][2]=mean1[i]
    set3[i][3]=ku_2[i]
    set3[i][4]=RMSA_2[i]
    set3[i][5]=ptp_2[i]
print(set3)


# In[9]:


print(set3[0])


# In[10]:


#把向量转化为Rs向量
import math
Rs0=1
fengzi=0
qq1=0
qq2=0
fengmu1=0
fengmu2=0
fengmu=0
Rst1=[]
for j in range(984):
    f0_avg=np.mean(set3[0])
    f1_avg=np.mean(set3[j])
    for i in range(6):
        qq1=set3[0][i]-f0_avg
        qq2=set3[j][i]-f1_avg
        fengzi=np.abs(fengzi+qq1*qq2)
        fengmu1=fengmu1+qq1*qq1
        fengmu2=fengmu2+qq2*qq2
    fengmu=fengmu1*fengmu2
    #print(fengmu)
    #print(fengzi)
    Rst1.append(fengzi/math.sqrt(fengmu1*fengmu2))
print(Rst1)


# In[12]:


#开始尝试进行小波包分解
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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
for i in range(984):
    plot_signal_decomp(set1[i][1], 'sym5', "DWT: Ecg sample - Symmlets5")
print(set4)


# In[13]:


print(len(set4[0]))


# In[14]:


#计算能量比
#分为8段：0-2560；2560-5120；5120-7680；7680-10240
#10240-12800;12800-15360;15360-17920;17920-20480
Et=[]
for i in range(984):
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


# In[15]:


#构建特征集
feature= [[0] * 14 for i in range(984)]
for i in range(984):
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
print(feature)


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
    for i in range(984):
        ss1.append(feature[i][j])
    Fb=np.mean(ss1)
    Lb=983/2
    for i in range(984):
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
    for i in range(984):
        ss2.append(feature[i][j])
    for i in range(983):
        if(ss2[i+1]-ss2[i]>0):
            Zeng=Zeng+1
        if(ss2[i+1]-ss2[i]<0):
            Fu=Fu+1
    Mon=np.abs((Zeng-Fu)/983)
    ZMon.append(Mon)
print(ZMon)


# In[18]:


#计算敏感特性的标准Cri=（Corr+Mon）/2
ZCri=[]
for i in range(14):
    ZCri.append((Zcorr[i]+ZMon[i])/2)
print(ZCri)


# In[ ]:




