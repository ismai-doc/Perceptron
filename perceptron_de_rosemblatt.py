#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[62]:


# Chargement des données
data = pd.read_csv('iris.csv')
data = data.drop(['Id'], axis=1)
data = data[:100]
data


# In[63]:


x = data.iloc[:100,[0, 2]].values
y = data.iloc[0:100, 4].values


# In[64]:


# convertion des variables en binaire
y = np.where(y == 'Iris-setosa', -1, 1)


# In[65]:


# séparation des données train et test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[66]:


# Modification d'echelle
sc_x = StandardScaler()
x_train_scale = sc_x.fit_transform(x_train) 
x_test_scale = sc_x.transform(x_test)


# In[68]:


# plot DES DONNEES
plt.scatter(x[:49, 0], x[:49, 1], color='red', marker='o', label='setosa')
plt.scatter(x[49:100, 0], x[49:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# ### L'IMPLEMENTATION D'ALGORITHME
# 

# In[74]:


def gradient_descent(alpha, x_train_scale, y_train, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x_train_scale.shape[0]
    
    w = np.random.random(x_train_scale.shape[1])
    b = np.random.random()
    
    dot = []
    x_l = []
    y_l = []
    for i in range(m):
        L = float((np.dot(w, x_train_scale[i:i+1,:].reshape(2,))+b)*y_train[i:i+1]) #+b
        if L < 0:
            dot.append(L)
            x_l.append(x_train_scale[i:i+1,:])
            y_l.append(y_train[i:i+1])
            
    len(dot)!= 0
    x_lo = []
    y_lo = []        
    loss = sum(dot)/len(dot)
    x_lo = np.array(x_l).reshape(len(dot),2)
    y_lo = np.array(y_l).reshape(len(dot),1)
    
    while not converged:
   # for each training sample, compute the gradient (d/d_theta j(theta))
        
        grad_w = 1/len(dot) * sum(-(x_lo*y_lo))
        grad_b = 1/len(dot) * float(sum(-(y_lo)))

        # update the theta_temp
        temp_w = w - alpha * grad_w
        temp_b = b - alpha * grad_b
        
        # update theta
        w = temp_w
        b = temp_b

        # loss with new weights
        dot1 = []
        x_l = []
        y_l = []
       #len(dot1) != 0 
        for i in range(m):
            E = float((np.dot(w, x_train_scale[i:i+1,:].reshape(2,))+b)*y_train[i:i+1]) #+b
            if E < 0:
                dot1.append(E)
                x_l.append(x_train_scale[i:i+1,:])
                y_l.append(y_train[i:i+1])
                 
        x_lo = []
        y_lo = []
        
        loss_upd = sum(dot1)/len(dot1)
        x_lo = np.array(x_l).reshape(len(dot1),2)
        y_lo = np.array(y_l).reshape(len(dot1),1)

        if abs(loss-loss_upd) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True
    
        loss = loss_upd   # update error 
        #x_l = x_l1
        #y_l = y_l1
        iter += 1  # update iter
    
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True
    
    return w,b


# In[96]:


alpha = 0.01 # learning rate
ep = 0.00005 # convergence criteria

# call gredient decent, and get intercept(=theta0) and slope(=theta1)
weights, bias = gradient_descent(alpha, x_train_scale, y_train, ep, max_iter=10000)
print(weights, bias)


# In[97]:


xx = np.linspace(-3, 3)
a = -weights[0]/weights[1]
yy = a*xx - bias/weights[1]

plt.figure(2, figsize=(8, 6))

# Plot the training points
plt.scatter(x_train_scale[:, 0], x_train_scale[:, 1], c=y_train, cmap='bwr', edgecolor='k')    #plt.cm.Set1,


plt.plot(xx, yy)
plt.ylim(-3,3)
plt.xlim(-3,3)

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.grid()
plt.show()


# #### DONNEES DE TEST
# 

# In[98]:


m_test = x_test_scale.shape[0]
dot_test = []
x_l = []
y_l = []
for i in range(m_test):
    L = float((np.dot(weights, x_test_scale[i:i+1,:].reshape(2,))+bias)*y_test[i:i+1]) #+b
    if L < 0:
        dot_test.append(L)
        x_l.append(x_test_scale[i:i+1,:])
        y_l.append(y_test[i:i+1])

print('accuracy:', 1-len(dot_test)/len(x_test_scale))


# In[102]:


xx = np.linspace(-3, 3)
a = -weights[0]/weights[1]
yy = a*xx - bias/weights[1]

plt.figure(2, figsize=(8, 6))

# Plot the test points
plt.scatter(x_test_scale[:, 0], x_test_scale[:, 1], c=y_test, cmap='bwr',    #plt.cm.Set1,
            edgecolor='k')

plt.plot(xx, yy)
plt.ylim(-3,3)
plt.xlim(-3,3)

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.grid()
plt.show()


# In[ ]:




