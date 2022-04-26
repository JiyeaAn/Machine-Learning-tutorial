#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
# statsmodels: 검정 및 추정 , 회귀분석, 시계열분석등의 다양한 통계분석 기능을 제공


# In[2]:


os.getcwd()


# In[3]:


boston = pd.read_csv("./data/Boston_house.csv")


# In[4]:


boston.head()


# In[5]:


# Target이 예측해야하는 값이기 때문에 제외하고 데이터 뽑기
boston_data = boston.drop(['Target'], axis=1)


# In[6]:


# data 통계 뽑아보기
boston_data.describe()


# #### 타겟 데이터
# - 978 보스턴 주택 가격
# - 506개 타운의 주택 가격 중앙값 (단위 1,000 달러)
# 
# #### 특징 데이터
# - CRIM: 범죄율
# - INDUS: 비소매상업지역 면적 비율
# - NOX: 일산화질소 농도
# - RM: 주택당 방 수
# - LSTAT: 인구 중 하위 계층 비율
# - B: 인구 중 흑인 비율
# - PTRATIO: 학생/교사 비율
# - ZN: 25,000 평방피트를 초과 거주지역 비율
# - CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
# - AGE: 1940년 이전에 건축된 주택의 비율
# - RAD: 방사형 고속도로까지의 거리
# - DIS: 직업센터의 거리
# - TAX: 재산세율

# In[7]:


# 변수 별로 각각 단순 선형 회귀 분석하기 (3개 변수만 해보기)
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]


# In[8]:


target


# In[9]:


# target ~ crim 선형회귀분석
# crim변수에 상수항 추가하기 : sm.add_constant
# 데이터 행이 하나인 경우에는 add_constant에 has_constant="add" 인수를 추가

crim1 = sm.add_constant(crim, has_constant="add")
crim1


# In[10]:


# sm.OLS 적합시키기
model1 = sm.OLS(target, crim1) # Y값부터 나와야함
fitted_model1 = model1.fit()


# In[12]:


# summary 함수를 통해서 결과 출력
fitted_model1.summary()


# #### R-squared 해석
# - 이 X가 설명하는 Y의 총 변동성은 약 15% 정도 된다.
# 
# #### coef (회귀 계수값) 해석
# - 범죄율이 1 단위 증가할 때, Y가 -0.4 단위 증가한다.(= 0.4 단위 감소)
# 
# #### P>|t| 해석 (P-value)
# - 매우 유의미하다.

# In[13]:


# 회귀 계수 출력
fitted_model1.params


# In[14]:


# 예측값: y_hat=beta0 + beta1 * X 계산해보기

# 회귀 계수 X 데이터(X)
np.dot(crim1, fitted_model1.params) # 벡터 내적, 행렬곱 함수인 np.dot 함수


# In[15]:


# predict함수를 통해 y_hat 구하기
pred1 = fitted_model1.predict(crim1)


# In[16]:


# 직접 구한 y_hat과 predict함수를 통해 구한 y_hat 차이
np.dot(crim1, fitted_model1.params) - pred1


# ### 적합시킨 직선 시각화

# In[17]:


import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(crim, target, label = "data")
plt.plot(crim, pred1, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()

## 음의 상관관계를 갖지만, 전반적으로 잘 맞는거 같지 않음


# In[18]:


plt.scatter(target, pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()

## 전반적으로 잘 맞는거 같지 않음


# In[19]:


# residual 시각화
fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()

# 잔차가 균일하지 않다! 
# 그러므로 보스턴하우스 집값을 예측함에 있어서 범죄율은 적합하지 않은거 같음


# In[21]:


# 잔차의 합계산해보기
np.sum(fitted_model1.resid)


# ### 다른 변수들 RUN

# In[25]:


# 상수항 추가
rm1 = sm.add_constant(rm, has_constant="add")
lstat1 = sm.add_constant(lstat, has_constant="add")


# In[26]:


# sm.OLS 적합시키기
model2 = sm.OLS(target, rm1)
fitted_model2 = model2.fit()
model3 = sm.OLS(target, lstat1)
fitted_model3 = model3.fit()


# In[27]:


fitted_model2.summary()


# In[28]:


fitted_model3.summary()


# In[30]:


# predict함수를 통해 y_hat 구하기
pred2 = fitted_model2.predict(rm1)
pred2


# In[31]:


pred3 = fitted_model3.predict(lstat1)
pred3


# In[32]:


# rm 모델 시각화
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(rm, target, label = "data")
plt.plot(rm, pred2, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()


# In[33]:


# lstat 모델 시각화
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(lstat, target, label = "data")
plt.plot(lstat, pred3, label = "result")
plt.legend() # 범례를 표시할건지
plt.show()


# In[34]:


# rm 모델 잔차
fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[35]:


# lstat 모델 잔차
fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[37]:


# 세 개의 모델 잔차 비교
fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()

# lstat가 가장 0 주변에 몰려있고, R-squared 값도 크기 때문에 영향이 크다고 할 수 있음

