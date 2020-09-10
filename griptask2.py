
# coding: utf-8

# # Simple Linear Regression
# 

#  # Importing the libraries

# In[2]:



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset 

# In[3]:


dataset = pd.read_csv('hscore.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# # Splitting the dataset into the Training set and Test set
# 
# 

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# # Training the Simple Linear Regression model on the Training set
# 

# In[8]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# # Predicting the Test set results

# In[11]:


y_pred = regressor.predict(X_test)
print(y_pred)


# # Visualising the Training set results

# In[12]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Score vs hour (Training set)')
plt.xlabel('hour')
plt.ylabel('Score')
plt.show()


# # Visualising the Test set results

# In[13]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Score vs Hour (Test set)')
plt.xlabel('Hour')
plt.ylabel('Score')
plt.show()


# # predicted score

# In[14]:


noh = 9.25
print("Number of hours : ",noh)
print("Predicted score : ",regressor.predict(np.array(noh).reshape(1,-1))[0])

