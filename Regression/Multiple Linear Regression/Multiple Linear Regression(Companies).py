#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


get_ipython().system('pip install statsmodels')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot


# # Importing Dataset 

# In[ ]:


dataset = pd.read_csv("50_Startups.csv") 


# In[ ]:


dataset


# In[ ]:


X = dataset.iloc[:,:-1].values


# In[ ]:


y = dataset.iloc[:,4].values


# ### Encoding Categorical Values

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


label_encoder_X = LabelEncoder()


# In[ ]:


X[:, 3] = label_encoder_X.fit_transform(X[:,3])


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [3])


# In[ ]:


X = onehotencoder.fit_transform(X).toarray()


# ### Avoiding Dummy Variable Trap

# In[ ]:


X = X[:,1:]


# # Splitting Dataset into test and train

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Fitting Multiple Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


regressor = LinearRegression()


# In[ ]:


regressor.fit(X_train, y_train)


# # Predicting the test set results

# In[ ]:


y_pred = regressor.predict(X_test)


# # Building Optimal model using Backward Elimination 

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Statsmodels does not contain b0 from MLR equation


# In[ ]:


# We need to add a column of 1s to make sure statsmodels recognises
# b0


# In[ ]:


X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


# In[ ]:


X_opt = X[:,[0,1,2,3,4,5]]


# In[ ]:


regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()


# In[ ]:


regressor_ols.summary()


# In[ ]:


X_opt = X[:, [0,1,3,4,5]]


# In[ ]:


regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()


# In[ ]:


regressor_ols.summary()


# In[ ]:


X_opt = X[:, [0,3,4,5]]


# In[ ]:


regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()


# In[ ]:


regressor_ols.summary()


# In[ ]:


X_opt = X[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()


# In[ ]:


# 0.000 means very, very small value


# In[ ]:


X_opt = X[:, [0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

