#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the Dataset 

# In[4]:


dataset = pd.read_csv("Data.csv")


# In[5]:


X = dataset.iloc[:,:-1].values


# In[6]:


# iloc[: = all rows, :-1 = all columns except last].values = all values


# In[7]:


X


# In[8]:


y = dataset.iloc[:,3].values


# In[9]:


y


# # Missing Data

# In[10]:


from sklearn.preprocessing import Imputer


# In[11]:


imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)


# In[12]:


# missing_values = identifying missing values in dataset
# strategy =  what to replace with
# axis = 0 for columns, 1 for rows


# In[13]:


imputer = imputer.fit(X[:, 1:3])


# In[14]:


# .fit(data[rows, columns])


# In[15]:


X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[16]:


# transform replaces values


# In[17]:


X 


# # Categorical Data

# In[18]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[19]:


labelencoder_X = LabelEncoder()


# In[20]:


X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# In[21]:


X


# ### This function transforms categorical values into encoded numerical values

# ### To make sure machine doesn't interpret these as being greater than or less than each other, we will use dummy encoding

# #### This ensures above condition is nullified, and the number of columns is equal to the number of labels

# In[22]:


onehotencoder = OneHotEncoder(categorical_features = [0])


# In[23]:


X = onehotencoder.fit_transform(X).toarray()


# In[24]:


X


# In[25]:


labelencoder_y = LabelEncoder()


# In[26]:


y = labelencoder_y.fit_transform(y)


# # Test and Train Split

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Feature Scaling

# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


sc_X = StandardScaler()


# In[31]:


X_train = sc_X.fit_transform(X_train)


# In[33]:


X_test = sc_X.transform(X_test)


# ##### Do we need to transform dummy variables or scale them?

# ##### It depends on the context, how much we wanna interpret our models. Won't break the model, but will damage interpretation.

# In[ ]:





# In[ ]:




