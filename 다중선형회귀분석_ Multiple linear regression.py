#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[2]:


os.getcwd()


# ## 단순선형회귀분석

# In[3]:


boston = pd.read_csv("./data/Boston_house.csv")


# In[4]:


boston.head()


# In[5]:


boston_data = boston.drop(['Target'], axis=1)


# In[6]:


# 변수 별로 각각 단순 선형 회귀 분석하기 (3개 변수만 해보기)
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]


# In[7]:


crim1 = sm.add_constant(crim, has_constant="add")
crim1


# In[8]:


model1 = sm.OLS(target, crim1) # Y값부터 나와야함
fitted_model1 = model1.fit()


# In[9]:


fitted_model1.summary()


# In[10]:


fitted_model1.params


# In[11]:


# 회귀 계수 X 데이터(X)
np.dot(crim1, fitted_model1.params)


# In[12]:


pred1 = fitted_model1.predict(crim1)


# In[13]:


np.dot(crim1, fitted_model1.params) - pred1


# In[14]:


import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(crim, target, label = "data")
plt.plot(crim, pred1, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()


# In[15]:


plt.scatter(target, pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()


# In[16]:


fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[17]:


np.sum(fitted_model1.resid)


# In[18]:


rm1 = sm.add_constant(rm, has_constant="add")
lstat1 = sm.add_constant(lstat, has_constant="add")


# In[19]:


model2 = sm.OLS(target, rm1)
fitted_model2 = model2.fit()
model3 = sm.OLS(target, lstat1)
fitted_model3 = model3.fit()


# In[20]:


fitted_model2.summary()


# In[21]:


fitted_model3.summary()


# In[22]:


pred2 = fitted_model2.predict(rm1)
pred2


# In[23]:


pred3 = fitted_model3.predict(lstat1)
pred3


# In[24]:


# rm 모델 시각화
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(rm, target, label = "data")
plt.plot(rm, pred2, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()


# In[25]:


# lstat 모델 시각화
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(lstat, target, label = "data")
plt.plot(lstat, pred3, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()


# In[26]:


# rm 모델 잔차
fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[27]:


# lstat 모델 잔차
fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[28]:


# 세 개의 모델 잔차 비교
fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()


# ## 다중선형회귀분석

# In[29]:


# boston data에서 crim, rm, lstat 변수만 뽑아오기
x_data = boston[['CRIM','RM','LSTAT']]
x_data.head()


# In[30]:


# 상수항추가
x_data1 = sm.add_constant(x_data, has_constant = "add")


# In[31]:


# 회귀모델 적합
multi_model = sm.OLS(target, x_data1)
fitted_multi_model1 = multi_model.fit()


# In[32]:


fitted_multi_model1.summary()


# ## 단순선형 회귀 분석과 다중 선형 회귀 분석 비교

# In[33]:


#단순 선형 회귀 모델의 회귀 계수
print(fitted_model1.params)
print(fitted_model2.params)
print(fitted_model3.params)


# In[34]:


print(fitted_multi_model1.params)
# 다중 공산성을 확인할 수 있음


# In[35]:


# 행렬 연산을 통해 beta 구하기 (X'X)-1X'y
from numpy import linalg # 역행렬을 위한 패키지
np.dot(x_data1.T, x_data1)


# In[36]:


ba = linalg.inv(np.dot(x_data1.T, x_data1))
np.dot(np.dot(ba, x_data1.T), target)


# In[38]:


# y_hat 구하기
pred4 = fitted_multi_model1.predict(x_data1)


# In[39]:


fitted_multi_model1.resid.plot()
plt.xlabel("residual_number")
plt.show


# In[41]:


fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
fitted_multi_model1.resid.plot(label="full")
plt.legend()
# 다중선형회귀분석의 잔차가 제일 작아짐 (당연히 변수가 많아지면 발생하는 결과)


# In[ ]:




