#!/usr/bin/env python
# coding: utf-8

# PCA+KNN

# In[27]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#对读取到的数据进行一维化扁平处理
def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

# 样本数据和样本标签
X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

#cnt用于计数
cnt = 0
#先读取训练集
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

#再重新读取测试集
for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        
#Max_point用于保存最优超参数模型的交叉验证评分
Max_point = 0
final_n = 0
final_k = 0

#该循环用于选出PCA超参数n_components的最佳值
#n_components: 需要保留的特征数量（即降维后的结果）
for n in range(10,410,10):
    #调用PCA
    pca = PCA(n_components = n) #实例化
    newX = pca.fit_transform(X) #用已有数据训练PCA模型，并返回降维后的数据

    #将降维后的数据拆分为训练集和测试集
    x_train = newX[0:320,:]
    y_train = Y[0:320]   
    x_test = newX[320:400,:]
    y_test = Y[320:400]

    #KNN的超参数：
    #n_neighbors：KNN用于判别分类的邻居数
    C = np.arange(3,22,2)
    #将需要遍历的超参数定义为字典
    params = {'n_neighbors': C}

    #定义网格搜索中使用的模型和参数
    knn = GridSearchCV(KNeighborsClassifier(), params, scoring = "f1_macro",cv = 5)
    #使用网格搜索模型拟合数据
    knn.fit(x_train,y_train)

    #存储PCA不同超参数下的KNN最优超参数模型的交叉验证评分
    cur_point = knn.best_score_
    #选出模型交叉验证评分最高的一组超参数（PCA和KNN）
    if cur_point > Max_point:
        Max_point = cur_point
        final_n = n
        final_k = knn.best_params_['n_neighbors']

#输出结果
print("\nPCA的超参数n_components的最优解为: %d\n" %final_n)
print("KNN的超参数n_neighbors的最优解为: %d\n" %final_k)
y_predict = knn.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+KNN

# In[13]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt=0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        

Max_Point = 0
final_n = 0
final_k = 0
for n in range(5,40):
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.arange(3,22,2)
    params = {'n_neighbors': C}

    knn = GridSearchCV(KNeighborsClassifier(), params, scoring = "f1_macro",cv = 5)

    knn.fit(x_train,y_train)

    cur_point = knn.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_k = knn.best_params_['n_neighbors']

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("KNN的超参数n_neighbors的最优解为: %d\n" %final_k)
y_predict = knn.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# KNN

# In[12]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt=0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        


final_k = 0
Max_point = 0

x_test = X[320:400,:]
y_test = Y[320:400]
x_train = X[0:320,:]
y_train = Y[0:320]

C = np.arange(3,22,2)
params = {'n_neighbors': C}

knn = GridSearchCV(KNeighborsClassifier(), params, scoring = "f1_macro",cv = 5)

knn.fit(x_train,y_train)

final_k = knn.best_params_['n_neighbors']

Max_point = knn.best_score_

print("KNN的超参数n_neighbors的最优解为: %d\n" %final_k)
y_predict = knn.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# PCA+SVM线性核

# In[14]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

Max_Point = 0
final_n = 0
final_C = 0
for n in range(10,410,10):
    pca = PCA(n_components = n)
    newX = pca.fit_transform(X)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(10))
    params = {'C': C,'kernel':['linear']}

    svc_linear = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_linear.fit(x_train,y_train)

    cur_point = svc_linear.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_linear.best_params_['C']

print("\nPCA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM线性核的超参数C的最优解为: %d\n" %final_C)
y_predict = svc_linear.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+SVM线性核

# In[15]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

Max_Point = 0
final_n = 0
final_C = 0
for n in range(5,40):
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(10))
    params = {'C': C,'kernel':['linear']}

    svc_linear = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_linear.fit(x_train,y_train)

    cur_point = svc_linear.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_linear.best_params_['C']

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM线性核的超参数C的最优解为: %d\n" %final_C)
y_predict = svc_linear.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# SVM线性核

# In[17]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

Max_Point = 0
final_C = 0



x_test = X[320:400,:]
y_test = Y[320:400]
x_train = X[0:320,:]
y_train = Y[0:320]

C = np.power(10, np.arange(10))
params = {'C': C,'kernel':['linear']}

svc_linear = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

svc_linear.fit(x_train,y_train)

final_C = svc_linear.best_params_['C']

Max_point = svc_linear.best_score_

print("SVM线性核的超参数C的最优解为: %d\n" %final_C)
y_predict = svc_linear.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# PCA+高斯内核rbf

# In[18]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt=0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        

Max_Point = 0
final_n = 0
final_C = 0
final_gamma = 0

for n in range(10,410,10):
    pca = PCA(n_components = n)
    newX = pca.fit_transform(X)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(10))
    params = {'C': C,'kernel':['rbf'],'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]}

    svc_rbf = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_rbf.fit(x_train,y_train)

    cur_point = svc_rbf.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_rbf.best_params_['C']
        final_gamma = svc_rbf.best_params_['gamma']


print("\nPCA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM高斯内核rbf的超参数C的最优解为: %d 超参数gamma的最优解为: %f\n" %(final_C,final_gamma))
y_predict = svc_rbf.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+高斯内核rbf

# In[22]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt=0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        

Max_Point = 0
final_n = 0
final_C = 0
final_gamma = 0

for n in range(5,40):
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(10))
    params = {'C': C,'kernel':['rbf'],'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]}

    svc_rbf = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_rbf.fit(x_train,y_train)

    cur_point = svc_rbf.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_rbf.best_params_['C']
        final_gamma = svc_rbf.best_params_['gamma']


print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM高斯内核rbf的超参数C的最优解为: %d 超参数gamma的最优解为: %f\n" %(final_C,final_gamma))
y_predict = svc_rbf.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# 高斯内核rbf

# In[21]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt=0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        


final_C = 0
final_gamma = 0
Max_point = 0




x_test = X[320:400,:]
y_test = Y[320:400]
x_train = X[0:320,:]
y_train = Y[0:320]

C = np.power(10, np.arange(10))
params = {'C': C,'kernel':['rbf'],'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]}

svc_rbf = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

svc_rbf.fit(x_train,y_train)

Max_point = svc_rbf.best_score_


final_C = svc_rbf.best_params_['C']
final_gamma = svc_rbf.best_params_['gamma']


print("SVM高斯内核rbf的超参数C的最优解为: %d 超参数gamma的最优解为: %f\n" %(final_C,final_gamma))
y_predict = svc_rbf.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# PCA+多项式内核

# In[23]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        

Max_Point = 0
final_n = 0
final_C = 0
final_gamma = 0
final_degree = 0
final_coef0 = 0
for n in range(10,410,10):
    pca = PCA(n_components = n)
    newX = pca.fit_transform(X)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(5))
    params = {'C': C,'kernel':['poly'],'gamma': [1e-1,1e-2,1e-3],'degree':[1,3,5],'coef0':[0,1,3,5]}

    svc_poly = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_poly.fit(x_train,y_train)

    cur_point = svc_poly.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_poly.best_params_['C']
        final_gamma = svc_poly.best_params_['gamma']
        final_degree = svc_poly.best_params_['degree']
        final_coef0 = svc_poly.best_params_['coef0']

print("\nPCA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM多项式内核poly的超参数C的最优解为: %d 超参数gamma的最优解为: %f 超参数degree的最优解为: %d 超参数coef0的最优解为: %d\n" %(final_C,final_gamma,final_degree,final_coef0))
y_predict = svc_poly.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+多项式内核

# In[25]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        

Max_Point = 0
final_n = 0
final_C = 0
final_gamma = 0
final_degree = 0
final_coef0 = 0
for n in range(5,40):
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    C = np.power(10, np.arange(5))
    params = {'C': C,'kernel':['poly'],'gamma': [1e-1,1e-2,1e-3],'degree':[1,3,5],'coef0':[0,1,3,5]}

    svc_poly = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

    svc_poly.fit(x_train,y_train)

    cur_point = svc_poly.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = svc_poly.best_params_['C']
        final_gamma = svc_poly.best_params_['gamma']
        final_degree = svc_poly.best_params_['degree']
        final_coef0 = svc_poly.best_params_['coef0']
        
print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("SVM多项式内核poly的超参数C的最优解为: %d 超参数gamma的最优解为: %f 超参数degree的最优解为: %d 超参数coef0的最优解为: %d\n" %(final_C,final_gamma,final_degree,final_coef0))
y_predict = svc_poly.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# 多项式内核

# In[26]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def img2vector(filename):
    img = mpimg.imread(filename) 
    return img.reshape(1, -1)

X = np.zeros((400,10304),dtype = int)
Y = np.zeros(400,dtype = int)

cnt = 0
for i in range(1,41):
    for j in range(1,9):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1

for i in range(1,41):
    for j in range(9,11):
        src = 'C:\\Users\\93508\\Desktop\\ORL\\s'
        src += str(i)
        src += '\\'
        src += str(j)
        src += '.bmp'
        X[cnt] = img2vector(src)
        Y[cnt] = i
        cnt = cnt + 1
        


final_C = 0
final_gamma = 0
final_degree = 0
final_coef0 = 0
Max_point = 0



x_test = newX[320:400,:]
y_test = Y[320:400]
x_train = newX[0:320,:]
y_train = Y[0:320]

C = np.power(10, np.arange(5))
params = {'C': C,'kernel':['poly'],'gamma': [1e-1,1e-2,1e-3],'degree':[1,3,5],'coef0':[0,1,3,5]}

svc_poly = GridSearchCV(SVC(), params, scoring = "f1_macro",cv = 5)

svc_poly.fit(x_train,y_train)

Max_point = svc_poly.best_score_


final_C = svc_poly.best_params_['C']
final_gamma = svc_poly.best_params_['gamma']
final_degree = svc_poly.best_params_['degree']
final_coef0 = svc_poly.best_params_['coef0']
        

print("SVM多项式内核poly的超参数C的最优解为: %d 超参数gamma的最优解为: %f 超参数degree的最优解为: %d 超参数coef0的最优解为: %d\n" %(final_C,final_gamma,final_degree,final_coef0))
y_predict = svc_poly.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为：%.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))

