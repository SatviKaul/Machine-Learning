#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the Dataset 

# In[2]:


dataset = pd.read_csv("Data.csv")


# In[3]:


X = dataset.iloc[:,:-1].values


# In[5]:


# iloc[: = all rows, :-1 = all columns except last].values = all values


# In[6]:


X


# In[7]:


y = dataset.iloc[:,3].values


# In[8]:


y


# # Missing Data

# In[9]:


from sklearn.preprocessing import Imputer


# In[10]:


imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)


# In[11]:


# missing_values = identifying missing values in dataset
# strategy =  what to replace with
# axis = 0 for columns, 1 for rows


# In[16]:


imputer = imputer.fit(X[:, 1:3])


# In[13]:


# .fit(data[rows, columns])


# In[15]:


X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[17]:


# transform replaces values


# In[18]:


X 


# # Categorical Data

# In[24]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[20]:


labelencoder_X = LabelEncoder()


# In[22]:


X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# In[23]:


X


# ### This function transforms categorical values into encoded numerical values

# ### To make sure machine doesn't interpret these as being greater than or less than each other, we will use dummy encoding

# #### This ensures above condition is nullified, and the number of columns is equal to the number of labels

# In[25]:


onehotencoder = OneHotEncoder(categorical_features = [0])


# In[26]:


X = onehotencoder.fit_transform(X).toarray()


# In[27]:


X


# In[28]:


labelencoder_y = LabelEncoder()


# In[29]:


y = labelencoder_y.fit_transform(y)


# # Test and Train Split

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[ ]:





# In[ ]:




