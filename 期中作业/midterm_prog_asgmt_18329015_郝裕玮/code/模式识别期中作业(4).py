#!/usr/bin/env python
# coding: utf-8

# LDA+逻辑回归

# In[5]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
for n in range(5,21):
    
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:].astype(int)
    y_test = Y[320:400].astype(int)
    x_train = newX[0:320,:].astype(int)
    y_train = Y[0:320].astype(int)

    C = np.power(10, np.arange(5)).astype(int)
    params = {'C': C,'solver':['liblinear']}

    LR = GridSearchCV(LogisticRegression(), params, scoring = "f1_macro",cv = 5)

    LR.fit(x_train,y_train)

    cur_point = LR.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_C = LR.best_params_['C']
        

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("逻辑回归的超参数C的最优解为: %d\n" %final_C)
y_predict = LR.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为: %.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+决策树

# In[8]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
from sklearn.tree import DecisionTreeClassifier
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
final_max_depth = 0
final_min_samples_split = 0
final_min_samples_leaf = 0

for n in range(5,21):
    
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    params = {'max_depth': [40,60,80],'min_samples_split':[2,4,6],'min_samples_leaf': [1,3,5]}

    dectree = GridSearchCV(DecisionTreeClassifier(), params, scoring = "f1_macro",cv = 5)

    dectree.fit(x_train,y_train)

    cur_point = dectree.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_max_depth = dectree.best_params_['max_depth']
        final_min_samples_split = dectree.best_params_['min_samples_split']
        final_min_samples_leaf = dectree.best_params_['min_samples_leaf']
        

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("决策树的超参数max_depth的最优解为: %d 超参数min_samples_split的最优解为: %d 超参数min_samples_leaf的最优解为: %d\n" %(final_max_depth,final_min_samples_split,final_min_samples_leaf))
y_predict = dectree.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为: %.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+随机森林

# In[10]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
final_n_estimators = 0
final_max_depth = 0

for n in range(5,21):
    
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    params = {'n_estimators':[100,300,500],'max_depth':[10,20,30]}

    rndtree = GridSearchCV(RandomForestClassifier(), params, scoring = "f1_macro",cv = 5)

    rndtree.fit(x_train,y_train)

    cur_point = rndtree.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_n_estimators = rndtree.best_params_['n_estimators']
        final_max_depth = rndtree.best_params_['max_depth']
        

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("随机森林的超参数n_estimators的最优解为: %d 超参数max_depth的最优解为: %d\n" %(final_n_estimators,final_max_depth))
y_predict = rndtree.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为: %.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+adaboost

# In[1]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
final_n_estimators = 0
final_learning_rate = 0

for n in range(5,21):
    
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    params = {'n_estimators':[100,300,500],'learning_rate':[0.3,0.6,0.9]}

    AB = GridSearchCV(AdaBoostClassifier(), params, scoring = "f1_macro",cv = 5)

    AB.fit(x_train,y_train)

    cur_point = AB.best_score_

    if cur_point > Max_Point:
        Max_Point = cur_point
        final_n = n
        final_n_estimators = AB.best_params_['n_estimators']
        final_learning_rate = AB.best_params_['learning_rate']
        

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("adaboost的超参数n_estimators的最优解为: %d 超参数learning_rate的最优解为: %.1f\n" %(final_n_estimators,final_learning_rate))
y_predict = AB.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为: %.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_Point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))


# LDA+神经网络

# In[13]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
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
        

Max_point = 0
final_n = 0
final_hidden_layer_sizes = 0
final_alpha = 0
final_max_iter = 0

for n in range(5,21):
    
    lda = LDA(n_components = n)
    newX = lda.fit_transform(X,Y)

    x_test = newX[320:400,:]
    y_test = Y[320:400]
    x_train = newX[0:320,:]
    y_train = Y[0:320]

    
    mlp_clf__tuned_parameters = {'hidden_layer_sizes':[10,30,50],'alpha':[0.1,0.001,0.001],'max_iter':[100,300,500]}
    
    mlp = GridSearchCV(MLPClassifier(), mlp_clf__tuned_parameters, scoring = "f1_macro", cv = 5)
    
    mlp.fit(x_train,y_train)

    cur_point = mlp.best_score_

    if cur_point > Max_point:
        Max_point = cur_point
        final_n = n
        final_hidden_layer_sizes = mlp.best_params_['hidden_layer_sizes']
        final_alpha = mlp.best_params_['alpha']
        final_max_iter = mlp.best_params_['max_iter']
        

print("\nLDA的超参数n_components的最优解为: %d\n" %final_n)
print("神经网络的超参数hidden_layer_sizes的最优解为: %d 超参数alpha的最优解为: %f 超参数max_iter的最优解为: %d\n" %(final_hidden_layer_sizes,final_alpha,final_max_iter))
y_predict = mlp.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print("测试集预测正确率为: %.2f%%\n" %(accuracy*100))
print("最优超参数模型的评分为: %.2f\n" %Max_point)
print("测试集的预测分类报告如下所示：\n\n")
print(classification_report(y_test, y_predict))

