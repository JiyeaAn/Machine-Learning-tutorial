#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[2]:


os.getcwd()


# ## 다중선형회귀분석

# In[3]:


boston = pd.read_csv("./data/Boston_house.csv")


# In[4]:


boston.head()


# In[5]:


boston_data = boston.drop(['Target'], axis=1)


# In[6]:


# boston data에서 crim, rm, lstat 변수만 뽑아오기
target = boston[['Target']]
x_data = boston[['CRIM','RM','LSTAT']]
x_data.head()


# In[7]:


# 상수항추가
x_data1 = sm.add_constant(x_data, has_constant = "add")


# In[8]:


# 회귀모델 적합
multi_model = sm.OLS(target, x_data1)
fitted_multi_model1 = multi_model.fit()


# In[9]:


fitted_multi_model1.summary()


# #### CRIM, RM, LSTAT, B, TAX, AGE, ZN, NOX, INDUS 변수를 통한 다중선형 회귀분석

# In[14]:


# boston data에서 원하는 변수만 뽑아오기
x_data2 = boston[["CRIM", "RM", "LSTAT", "B", "TAX", "AGE", "ZN", "NOX", "INDUS"]]
x_data2.head()


# In[15]:


# 상수항추가
x_data2_ = sm.add_constant(x_data2, has_constant="add")


# In[16]:


# 회귀모델 적합
multi_model2 = sm.OLS(target, x_data2_)
fitted_multi_model2 = multi_model2.fit()


# In[17]:


# 결과 출력
fitted_multi_model2.summary()


# In[18]:


# 3개의 변수만 추가한 모델의 회귀 계수
fitted_multi_model1.params


# In[19]:


# 모든 변수를 추가한 모델의 회귀 계수
# Y 변동성을 서로 중첩하게 갖지 못하므로 서로 빼앗긴다.(다중공선성)
fitted_multi_model2.params


# 만약 지금 같은 상황에서 fitted_multi_model1과 fitted_multi_model2 중에 모델을 선택해야한다면, fitted_multi_model1을 선택해야함
# - Adj. R-squared 즉, 비슷한 성능을 보이고 있고 각 모델이 가진 파라메타의 수가 차이가 많이 난다면 파라메터의 수가 적은 모델이 더 훨씬 좋은 모델임
# - 왜냐하면 학습하는 속도나 관리자의 입장에서나 좋음
# - NOX와 INDUS는 현재 P-value값도 높고 R-Squared 값도 높음 (유의미하지 않는 변수로 보임) > 이런 경우, 이 두개의 값을 제외하면 다른 변수들의 P-value 값이 낮아짐. (겹쳐지는 부분의 값이 제외된게 없어지므로)
# - CRIM은 애매한 상태인데, 분석자의 주관을 넣어서 판단해야함 (가치 판단)
# - 일반적으로 범죄율이 집값에 영향을 미치긴 하니, 넣으면 됨
# - NOX와 INDUS를 제거해보고 CRIM을 넣을지 뺄지 고민하는 순서로 분석함.

# In[21]:


import matplotlib.pyplot as plt
fitted_multi_model1.resid.plot(label="full")
fitted_multi_model2.resid.plot(label="full_add")
plt.legend()


# #### 상관계수/산점도를 통해 다중공선성 확인

# In[22]:


# 상관행렬

x_data2.corr()


# In[23]:


# 상관행렬 시각화 해서 보기
import seaborn as sns;
cmap = sns.light_palette("darkgray", as_cmap = True)
sns.heatmap(x_data2.corr(), annot=True, cmap=cmap)
plt.show()


# In[24]:


# 변수별 산점도 시각화
sns.pairplot(x_data2)
plt.show()


# #### VIF를 통한 다중공선성 확인

# In[29]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x_data2.values, i) for i in range(x_data2.shape[1])]
vif["features"] = x_data2.columns
vif


# In[33]:


# nox 변수 제거후 (x_data3) VIF 확인
# VIF 는 그 변수를 Y로 두고 나머지 변수로 회귀를 돌렸을 때, 나머지 변수로 그 변수를 설명하는 변동의 비율
# VIF가 10 이상이면 다중공선성이 있다고 진단을 하게됨

vif = pd.DataFrame()
x_data3 = x_data2.drop('NOX', axis = 1)
vif["VIF Factor"] = [variance_inflation_factor(x_data3.values, i) for i in range(x_data3.shape[1])]
vif["features"] = x_data3.columns
vif

#VIF가 높다고 무조건 지우는 것이 아닌, 상관관계와 P-value 등을 종합적으로 봐서 멈출지를 결정


# In[34]:


# RM 변수 제거후 (x_data4) VIF 확인
# VIF가 높다고 무조건 지우는 것이 아닌, 상관관계와 P-value 등을 종합적으로 봐서 멈출지를 결정
# VIF가 10 이상이면 다중공선성이 있다고 진단을 하게됨

vif = pd.DataFrame()
x_data4 = x_data3.drop('RM', axis = 1)
vif["VIF Factor"] = [variance_inflation_factor(x_data4.values, i) for i in range(x_data4.shape[1])]
vif["features"] = x_data4.columns
vif


# In[37]:


# NOX 변수 제거한 데이터 상수항 추가 후 회귀 모델 적합
# NOX, RM 변수 제거한 데이터 상수항 추가 후 회귀 모델 적합
x_data3_ = sm.add_constant(x_data3, has_constant= "add")
x_data4_ = sm.add_constant(x_data4, has_constant= "add")
model_vif = sm.OLS(target, x_data3_)
fitted_model_vif = model_vif.fit()
model_vif2 = sm.OLS(target, x_data4_)
fitted_model_vif2 = model_vif2.fit()


# In[38]:


# 회귀 모델 결과 비교
fitted_model_vif.summary()


# In[39]:


fitted_model_vif2.summary()


# ### 학습 / 검증데이터 분할

# In[47]:


from sklearn.model_selection import train_test_split
X = x_data2_
y = target
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


# In[49]:


# train_x 상수항 추가후 회귀 모델 적합
train_x.head()
fit_1 =  sm.OLS(train_y, train_x)
fit_1 = fit_1.fit()


# In[51]:


# 검증데이터에 대한 예측값과 true 값 비교
plt.plot(np.array(fit_1.predict(test_x)), label = "pred")
plt.plot(np.array(test_y), label = "true")
plt.legend()
plt.show()
# 패턴은 좀 찾아가고 있구나~!


# In[52]:


# x_data3와 x_data4 학습 검증 데이터 분할
X = x_data3_
y = target
train_x2, test_x2, train_y2, test_y2 = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)


# In[53]:


X = x_data4_
y = target
train_x3, test_x3, train_y3, test_y3 = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)


# In[54]:


test_y.head()


# In[55]:


test_y2.head()


# In[56]:


test_y3.head()


# In[58]:


# x_data3와 x_data4 회귀 모델 적합(fit2, fit3)
fit_2 =  sm.OLS(train_y2, train_x2)
fit_2 = fit_2.fit()
fit_3 =  sm.OLS(train_y3, train_x3)
fit_3 = fit_3.fit()


# In[60]:


# 검증데이터에 대한 예측값과 true 값 비교
plt.plot(np.array(fit_2.predict(test_x2)), label = "pred1")
plt.plot(np.array(fit_3.predict(test_x3)), label = "pred2")
plt.plot(np.array(test_y2), label = "true")
plt.legend()
plt.show()


# In[61]:


plt.plot(np.array(fit_1.predict(test_x)), label = "pred")
plt.plot(np.array(fit_2.predict(test_x2)), label = "pred_vif")
plt.plot(np.array(fit_3.predict(test_x3)), label = "pred_vif2")
plt.plot(np.array(test_y2), label = "true")
plt.legend()
plt.show()


# In[64]:


plt.plot(np.array(test_y2['Target']-fit_1.predict(test_x)),label="pred_full")
plt.plot(np.array(test_y2['Target']-fit_2.predict(test_x2)),label="pred_vif")
plt.plot(np.array(test_y2['Target']-fit_3.predict(test_x3)),label="pred_vif2")
plt.legend()
plt.show()


# ### MSE를 통한 검증데이터에 대한 성능비교

# In[66]:


# MSE는 작으면 작을수록 좋다
from sklearn.metrics import mean_squared_error


# In[70]:


mean_squared_error(y_true = test_y['Target'], y_pred = fit_1.predict(test_x))


# In[71]:


mean_squared_error(y_true = test_y['Target'], y_pred = fit_2.predict(test_x2))


# In[72]:


mean_squared_error(y_true = test_y['Target'], y_pred = fit_3.predict(test_x3))

