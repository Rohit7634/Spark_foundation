#!/usr/bin/env python
# coding: utf-8

# # importing  all libraries for this task

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading data from CSV file

# In[5]:


data=pd.read_csv('score.csv')
print("Data imported successfully")

data.head(5)


# # ploting data

# In[4]:


data.plot(x='Hours', y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[8]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[17]:


from sklearn.model_selection import train_test_split 
x_train, x_test,y_train,y_test=train_test_split(x,y,
                                                test_size=0.2, random_state=0)


# In[13]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

print("training complete.")


# # ploting the regression line through scatter plot

# In[15]:


line =regressor.coef_*x+regressor.intercept_

plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[24]:


print(x_test)
y_pred=regressor.predict(x_test)


# In[25]:


df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)


# In[30]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[ ]:




