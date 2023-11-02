#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[36]:


data = pd.read_csv('/Users/kadiresindhureddy/Downloads/Housing.csv')


# In[37]:


print(data.head()) 


# In[38]:


data['mainroad'] = data['mainroad'].apply(lambda x: 1 if x == 'yes' else 0)
data['guestroom'] = data['guestroom'].apply(lambda x: 1 if x == 'yes' else 0)
data['basement'] = data['basement'].apply(lambda x: 1 if x == 'yes' else 0)
data['hotwaterheating'] = data['hotwaterheating'].apply(lambda x: 1 if x == 'yes' else 0)
data['airconditioning'] = data['airconditioning'].apply(lambda x: 1 if x == 'yes' else 0)
data['prefarea'] = data['prefarea'].apply(lambda x: 1 if x == 'yes' else 0)


# In[39]:


data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})


# In[40]:


X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = data['price']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[43]:


y_pred = model.predict(X_test)


# In[44]:


output = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(output)


# In[45]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[46]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared (R^2): {r2}")


# In[34]:


# Plotting the predicted vs. actual house prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', label='Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Actual Prices')
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.legend()
plt.title("Actual vs. Predicted House Prices")
plt.show()


# In[ ]:




