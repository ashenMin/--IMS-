
# coding: utf-8

# In[1]:


#from sklearn import 
import pandas as pd
import numpy as np
import os

#导入set1数据到set1
os.chdir('D:/UNIVERSITY_STUDY/PHM/2nd_term/NASA PHM/IMS/2nd_test/2nd_test/')
file_chdir = os.getcwd()
set1 = []
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        File=pd.read_table(file,sep='\s+',header=None,engine='python')
        set1.append(File)


# In[2]:


set1[1]


# In[3]:


from pylab import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


# In[4]:


#time=['0','5000','10000','15000','20000','25000']
plt.figure(figsize=(12,3))
x1 = range(20480)
y1 = set1[0][:][1]
plt.plot(x1, y1, marker='.', mec='r', mfc='w')
plt.legend()
#plt.xticks(x1,time,rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.30)
plt.xlabel(u"sampling point") #X轴标签
plt.ylabel("amplitude") #Y轴标签
plt.title("extraction time 1") #标题
plt.show()


# In[5]:


set2=[]
plt.figure(figsize=(22,6))  #利用快速傅立叶变换将2号轴承时域转化为频域
x2 = np.linspace(0,1,20480)
i=0
for i in range(984):
    y2 = set1[i][:][1]
    yy = fft(y2)
    yf1=abs(fft(y2))/len(x2)           #归一化处理
    yf2 = yf1[range(int(len(x2)/2))]  #由于对称性，只取一半区间
    xf1 = np.arange(len(y2))        # 频率
    xf2 = xf1[range(int(len(x2)/2))]  #取一半区间
    set2.append(yf2)

plt.subplot(221)                     #选取一个文件的数据来展示
plt.plot(xf2,yf2,'r')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
plt.show()


# In[7]:


from sklearn import *
import numpy as np
from scipy import stats

#提取2号轴承均方根误差RMSE
i = 0
rmse = []
Frmse = []
x = range(20480)
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    yf = set2[i]
    rmse.append(np.sqrt(metrics.mean_squared_error(x, y)))
    Frmse.append(np.sqrt(metrics.mean_squared_error(xf, yf)))

plt.figure(figsize=(15,6))
plt.subplot(221)    
plt.plot(xs,rmse,'r')
plt.xlabel(u"Time domain-RMSE") #X轴标签
plt.ylabel("RMSE/g") #Y轴标签
plt.subplot(222)
plt.plot(xs,Frmse,'r')
plt.xlabel(u"Freq domain-RMSE") #X轴标签
plt.ylabel("RMSE/g") #Y轴标签
plt.show()


# In[8]:


#提取2号轴承绝对均值
i = 0
j = 0
k = 0
mean_2 = []
Fmean_2 = []
x = range(20480)
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    yf = set2[i]
    y1 = np.array([y]).T
    y2 = np.array([yf]).T
    for j in range(20480):
        y1[j]=abs(y1[j])
    for k in range(10240):
        y2[k]=abs(y2[k])
    mean_2.append(np.mean(y1))
    Fmean_2.append(np.mean(y2))

plt.figure(figsize=(15,6))
plt.subplot(221)
plt.plot(xs,mean_2,'r')
plt.xlabel(u"Time domain-mean") #X轴标签
plt.ylabel("mean/g") #Y轴标签
plt.subplot(222)
plt.plot(xs,Fmean_2,'r')
plt.xlabel(u"Freq domain-mean") #X轴标签
plt.ylabel("mean/g") #Y轴标签
plt.show()


# In[9]:


#提取2号轴承方差VAR
i = 0
var2 = []
Fvar2 = []
x = range(20480)
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    yf = set2[i]
    var2.append(np.var(y))
    Fvar2.append(np.var(yf))

plt.figure(figsize=(15,6))
plt.subplot(221)
plt.plot(xs,var2,'r')
plt.xlabel(u"Time domain-VAR") #X轴标签
plt.ylabel("VAR") #Y轴标签
plt.subplot(222)
plt.plot(xs,Fvar2,'r')
plt.xlabel(u"Freq domain-VAR") #X轴标签
plt.ylabel("VAR") #Y轴标签
plt.show()


# In[10]:


#提取2号轴承峰度
i = 0
j = 0
ku_2 = []
Fku_2 = []
x = range(20480)   
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    yf = set2[i]
    ku_2.append(stats.kurtosis(y))
    Fku_2.append(stats.kurtosis(yf))

plt.figure(figsize=(15,6))
plt.subplot(221)
plt.plot(xs,ku_2,'r')
plt.xlabel(u"Time domain-ku") #X轴标签
plt.ylabel("ku") #Y轴标签
plt.subplot(222)
plt.plot(xs,Fku_2,'r')
plt.xlabel(u"Freq domain-ku") #X轴标签
plt.ylabel("ku") #Y轴标签
plt.show()


# In[11]:


#提取2号轴承方根幅值
i = 0
j = 0
k = 0
RMSA_2 = []
F_RMSA_2 = []
x = range(20480)   
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    y1 = np.array([y]).T
    yf = set2[i]
    yf1 = np.array([yf]).T
    for j in range(20480):
        y1[j]=abs(y1[j])
    for k in range(10240):
        yf1[k]=abs(yf1[k])
    RMSA_2.append(np.square(sum(np.sqrt(y1) / 20480)))
    F_RMSA_2.append(np.square(sum(np.sqrt(yf1) / 10240)))

plt.figure(figsize=(15,6))
plt.subplot(221)
plt.plot(xs,RMSA_2,'r')
plt.xlabel(u"Time domain-RMSA") #X轴标签
plt.ylabel("RMSA") #Y轴标签
plt.subplot(222)
plt.plot(xs,F_RMSA_2,'r')
plt.xlabel(u"Freq domain-RMSA") #X轴标签
plt.ylabel("RMSA") #Y轴标签
plt.show()


# In[12]:


#提取2号轴承峰峰值
i = 0
j = 0
ptp_2 = []
Fptp_2 = []
x = range(20480)  
xf = range(10240)
xs = range(984)
for i in range(984):
    y = set1[i][:][1]
    yf = set2[i]
    ptp_2.append(max(y) - min(y))
    Fptp_2.append(max(yf) - min(yf))

plt.figure(figsize=(15,6))
plt.subplot(221)
plt.plot(xs,ptp_2,'r')
plt.xlabel(u"Time domain-ptp") #X轴标签
plt.ylabel("peak-to-peak value") #Y轴标签
plt.subplot(222)
plt.plot(xs,Fptp_2,'r')
plt.xlabel(u"Freq domain-ptp") #X轴标签
plt.ylabel("peak-to-peak value") #Y轴标签
plt.show()


# In[15]:


#整合6项时域特征生成新的2号轴承特征数据集
newData = [[0] * 12 for i in range(984)]
dataLabel = [0 for i in range(984)]
for i in range(984):
    newData[i][0] = rmse[i]
    newData[i][1] = mean_2[i]
    newData[i][2] = var2[i]
    newData[i][3] = ku_2[i]
    newData[i][4] = RMSA_2[i]
    newData[i][5] = ptp_2[i]
    newData[i][6] = Frmse[i]
    newData[i][7] = Fmean_2[i]
    newData[i][8] = Fvar2[i]
    newData[i][9] = Fku_2[i]
    newData[i][10] = F_RMSA_2[i]
    newData[i][11] = Fptp_2[i]
    #if i <= 700:
    #    dataLabel[i] = 0 #正常状态
    #elif i >700 and i <= 900:
    #    dataLabel[i] = 1 #轻中度磨损
    #else:
    #    dataLabel[i] = 2 #重度磨损


# In[18]:


from sklearn.decomposition import PCA  #建立PCA模型
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier  #神经网络
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# 标准化
#sc = StandardScaler()
#scData = sc.fit_transform(newData)
scData = scale(newData)
# 分割数据的特征向量和标记
X_digits = scData      #得到12位特征值
#y_digits = dataLabel       #得到对应的标签

# PCA降维：降到2维
estimator = PCA(n_components=2)
X_pca=estimator.fit_transform(X_digits)


# In[17]:


X_pca[:][:]


# In[19]:


#用Kmeans进行无监督聚类
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X_pca)#聚类
datalabel = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[datalabel == 0]
x1 = X[datalabel == 1]
x2 = X[datalabel == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


# In[18]:


# 显示轴承3个时期特征经PCA压缩后的2维空间分布
colors = ['black', 'blue', 'yellow']
fig = plt.figure()
axes = fig.add_subplot(111)
for i in range(len(dataLabel)):
    if dataLabel[i] == 0:
        axes.scatter(X_pca[i][0], X_pca[i][1], c=colors[0])#正常状态标黑色
    if dataLabel[i] == 1:
        axes.scatter(X_pca[i][0], X_pca[i][1], c=colors[1])#轻中度磨损标蓝色
    if dataLabel[i] == 2:
        axes.scatter(X_pca[i][0], X_pca[i][1], c=colors[2])#重度磨损标黄色
axes.legend(labels = ['0', '1', '2'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# In[19]:


print(estimator.explained_variance_ratio_)  #累计贡献率


# In[20]:


#初步测试pca降维分类准确性
pcax_train, pcax_test, pcay_train, pcay_test = train_test_split(X_pca, dataLabel, test_size=0.3, random_state=42) #随机分割训练集，测试集
model1 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
model1.fit(pcax_train, pcay_train)
#predict=model1.predict(pcax_test)
scorePca = model1.score(pcax_test, pcay_test)
scorePca


# In[22]:


from sklearn.decomposition import KernelPCA  #建立KPCA模型

kpca = KernelPCA(n_components=3, kernel="linear",gamma=11) #选择linear线性核函数
x_kpca1 = kpca.fit_transform(scData)

# 显示轴承3个时期特征经KPCA压缩后的2维空间分布
colors = ['black', 'blue', 'yellow']
fig = plt.figure()
axes = fig.add_subplot(111)
for i in range(len(dataLabel)):
    if dataLabel[i] == 0:
        axes.scatter(x_kpca1[i][0], x_kpca1[i][1], c=colors[0])
    if dataLabel[i] == 1:
        axes.scatter(x_kpca1[i][0], x_kpca1[i][1], c=colors[1])
    if dataLabel[i] == 2:
        axes.scatter(x_kpca1[i][0], x_kpca1[i][1], c=colors[2])
axes.legend(labels = ['0', '1', '2'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# In[23]:


#初步测试核函数为linear的kpca降维分类准确性
#随机分割kpca降维数据为训练集，测试集
kpcaX_train, kpcaX_test, kpcaY_train, kpcaY_test = train_test_split(x_kpca1, dataLabel, test_size=0.3, random_state=42) 
model2 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
model2.fit(kpcaX_train, kpcaY_train)
#predict=model1.predict(pcax_test)
scoreKpca1 = model2.score(kpcaX_test, kpcaY_test)
scoreKpca1


# In[34]:


kpca = KernelPCA(n_components=3, kernel="poly",gamma=11) #选择poly核函数
x_kpca2 = kpca.fit_transform(scData)

# 显示轴承3个时期特征经KPCA压缩后的2维空间分布
colors = ['black', 'blue', 'yellow']
fig = plt.figure()
axes = fig.add_subplot(111)
for i in range(len(dataLabel)):
    if dataLabel[i] == 0:
        axes.scatter(x_kpca2[i][0], x_kpca2[i][1], c=colors[0])
    if dataLabel[i] == 1:
        axes.scatter(x_kpca2[i][0], x_kpca2[i][1], c=colors[1])
    if dataLabel[i] == 2:
        axes.scatter(x_kpca2[i][0], x_kpca2[i][1], c=colors[2])
axes.legend(labels = ['0', '1', '2'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# In[37]:


#初步测试核函数为poly的kpca降维分类准确性
#随机分割kpca降维数据为训练集，测试集
kpcaX_train, kpcaX_test, kpcaY_train, kpcaY_test = train_test_split(x_kpca2, dataLabel, test_size=0.3, random_state=42) 
model3 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
model3.fit(kpcaX_train, kpcaY_train)
#predict=model1.predict(pcax_test)
scoreKpca2 = model3.score(kpcaX_test, kpcaY_test)
scoreKpca2


# In[39]:


kpca = KernelPCA(n_components=3, kernel="rbf",gamma=11) #选择rbf（径向基函数，Radial Basis Function）核函数
x_kpca3 = kpca.fit_transform(scData)

# 显示轴承3个时期特征经KPCA压缩后的2维空间分布
colors = ['black', 'blue', 'yellow']
fig = plt.figure()
axes = fig.add_subplot(111)
for i in range(len(dataLabel)):
    if dataLabel[i] == 0:
        axes.scatter(x_kpca3[i][0], x_kpca3[i][1], c=colors[0])
    if dataLabel[i] == 1:
        axes.scatter(x_kpca3[i][0], x_kpca3[i][1], c=colors[1])
    if dataLabel[i] == 2:
        axes.scatter(x_kpca3[i][0], x_kpca3[i][1], c=colors[2])
axes.legend(labels = ['0', '1', '2'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# In[41]:


#初步测试核函数为rbf的kpca降维分类准确性
#随机分割kpca降维数据为训练集，测试集
kpcaX_train, kpcaX_test, kpcaY_train, kpcaY_test = train_test_split(x_kpca3, dataLabel, test_size=0.3, random_state=42) 
model4 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
model4.fit(kpcaX_train, kpcaY_train)
#predict=model4.predict(pcax_test)
scoreKpca3 = model4.score(kpcaX_test, kpcaY_test)
scoreKpca3


# In[42]:


#随机分割原始特征数据集为训练集，测试集
x_train, x_test, y_train, y_test = train_test_split(newData, dataLabel, test_size=0.3, random_state=42) 
model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
model.fit(x_train, y_train)
#predict=model.predict(pcax_test)
score_ori = model.score(x_test, y_test)

print("原始特征数据集分类准确度：%10.3f " % (score_ori*100) + "%")
print("pca降维后分类准确度：%10.3f " % (scorePca*100) + "%")
print("kpca（linear核）降维后分类准确度：%10.3f " % (scoreKpca1*100) + "%")
print("kpca（poly核）降维后分类准确度：%10.3f " % (scoreKpca2*100) + "%")
print("kpca（rbf核）降维后分类准确度：%10.3f " % (scoreKpca3*100) + "%")

