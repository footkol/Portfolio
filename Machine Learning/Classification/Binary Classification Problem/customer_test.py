#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests
import json


# In[6]:


url = 'http://localhost:9696/predict'


# In[10]:


# Generating random customer to test the model

customer = {
"gender": "female",
 "car_owner": "Y",
 "propert_owner": "Y",
 "children": 0,
 "type_income": "Commercial associate",
 "education":"Higher education",
 "marital_status": "Married",
 "housing_type": "House / apartment",
 "birthday_count": -13557,
 "work_phone": 0,
 "phone": 1,
 "email_id": 1,
 "type_occupation": "Managers",
 "family_members": 2,
 "employed_days": -2418,
 "Annual_income": 95850
}


# In[11]:


response = requests.post(url, json=customer) 
response


# In[12]:


result = response.json()
result


# In[13]:


print(json.dumps(result, indent=2))


# In[ ]:




