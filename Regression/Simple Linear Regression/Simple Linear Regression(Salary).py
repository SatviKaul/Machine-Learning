#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

# In[2]:


dataset= pd.read_csv("Salary_Data.csv")


# In[3]:


dataset.head


# In[5]:


X = dataset.iloc[:,:-1].values


# In[6]:


Y = dataset.iloc[:, 1].values


# # Splitting data into test and train

# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# #### For SLR, we don't need to take care of feature scaling, libraries do that for us

# # Fitting SLR on the Training Set

# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


regressor = LinearRegression()


# In[11]:


regressor.fit(X_train, Y_train)


# # Predicting the Test set results 

# In[12]:


y_pred = regressor.predict(X_test)


# # Visualising the Training set results 

# In[24]:


plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Training set)")
plt.xlabel = ("Years of Experience")
plt.ylabel = ("Salary")
plt.show()


# In[25]:


plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Test set)")
plt.xlabel = ("Years of Experience")
plt.ylabel = ("Salary")
plt.show()


# In[ ]:




