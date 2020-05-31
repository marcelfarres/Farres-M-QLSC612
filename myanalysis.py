#!/usr/bin/env python
# coding: utf-8

# # Load the data and explore it

# Load Data

# In[1]:


# Load the dataset

import os

import pandas as pd
from pandas import plotting as pdplt

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

from itertools import combinations 

from itertools import starmap 
  

csvf = '.'+ os.path.sep +'data'+os.path.sep+ 'brainsize.csv'

data = pd.read_csv(csvf, sep=';', index_col=0, na_values=".")
data.head()


# 
# 
# Check general stadistics of our data.

# In[2]:


data.describe()


# Let's check what data types we have 

# In[3]:


data.dtypes


# Data typese seems to make sense so we will not asert any of them for now 
# 
# We don't want to have NaN values in our data, before continuing, we will delete all of them.

# In[4]:


dataf = data.dropna()
print('We lost a total of ', data.shape[0] - dataf.shape[0], 'subjects')


# We did not lose may subjects, so we will igonre any nan.

# In[5]:


data = dataf


# # Create our first very important variable base on very fine observations

# One of our most interesting data we did not include in the previous csv was the Time to Get from the Door to the Chair (TGDC)  and also very intersting Time to Get from the Chair to the Door (TGCD).
# 
# Both are measured in seconds and are rounded to full seconds
# 
# These where calculated using the hacked security system of our lab. 

# Generate my for sure very interesting and correlated extra variable

# In[6]:


distributions = [
    {"type": np.random.normal, "kwargs": {"loc": 80, "scale": 40}},
    {"type": np.random.normal, "kwargs": {"loc": 150, "scale": 30}},
    {"type": np.random.normal, "kwargs": {"loc": 300, "scale": 30}},
]
coefficients = np.array([0.4, 0.6, .2])
coefficients /= coefficients.sum()      # in case these did not add up to 1
sample_size = data.shape[0]

our_seed = 10000
np.random.seed(our_seed)
num_distr = len(distributions)
rdata = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    rdata[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
partY = rdata[np.arange(sample_size), random_idx]

our_seed = 9000000
np.random.seed(our_seed)
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
partY2 = rdata[np.arange(sample_size), random_idx]

# plt.hist(partY, density=True)
# plt.show()

# plt.hist(partY2, density=True)
# plt.show()


# enbedd our new super amazin varaliable in our dataframe
# 

# In[7]:


data.insert(2, "TGDC", partY, False) 
data.insert(2, "TGCD", partY2, False) 


# Check that is inserted correctly

# In[8]:


data.describe()


# # Get an overview of our data 

# In[9]:


fig = plt.figure(figsize=(12,8))
pdplt.scatter_matrix(data[['Weight', 'Height', 'FSIQ', 'VIQ', 'PIQ', 'MRI_Count', 'TGDC', 'TGCD']], diagonal='kde', figsize=(12,12));


# Looking at the distributions we see that exploring Weight, MRI_Count and Height could be a good direction

# In[10]:



sns.set(style="ticks")

to_explore = ['PIQ', 'VIQ', 'FSIQ', 'Weight', 'MRI_Count', 'Height']

f, axes = plt.subplots(len(to_explore), 1, figsize=(10,12))
ptl_idx = 0
f.tight_layout(pad=3.0)

for varN in to_explore:
    sns.regplot(data[varN], data.TGDC, ax=axes[ptl_idx])
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[varN], data.TGDC)
    print('Looking at {:<10}  Slope:{:<10.3} P-value:{:<10.3}  R-squared:{:<10.3} '.format(varN, slope , r_value, r_value**2))
    ptl_idx += 1


# It seems that if we get rid of some big outliers we will be able to have a better result.
# 
# Also we confirm that Weight, MRI_Count and Height can be a good start.
# 
# We need to explore more our data.

# In[11]:


conditions_base = ['Weight', 'Height', 'FSIQ', 'VIQ', 'PIQ', 'MRI_Count', 'Gender']

comb_size = list(range(1, len(conditions_base)))

conditions = []

for v in comb_size:
    conditions.extend(combinations(conditions_base, v))

possible_op = [':', '+', '*']

min_pvalue = 0.05

our_super_models = []

for val_cond in conditions:
    
    size_op = len(val_cond)-1
    
    if size_op >= 1:
        op = combinations(possible_op, size_op )
        for op_val in op:
            super_model = 'TGDC ~' + ''.join([' '+ a+ ' ' + b for a,b in zip(val_cond[:-1],op_val)]) +' '+val_cond[-1]
        
            # Fit the model
            model= ols(super_model, data)
            model_fit = model.fit()
            
            is_valid = all( val < min_pvalue for idx, val in model_fit.pvalues.items())
            if is_valid:
                our_super_models.append(model)
                print(super_model)
                print(model_fit.summary())
        
    else:
        super_model = 'TGDC ~ {}'.format(*val_cond)
        model= ols(super_model, data)
        model_fit = model.fit()
            
        is_valid = all( val < min_pvalue for idx, val in model_fit.pvalues.items())
        if is_valid:
            our_super_models.append(model)
            print(super_model)
            print(model_fit.summary())

        


# We found 2 super interesting models and it makes a lot of sense
# 
# `TGDC ~ Weight : Height + MRI_Count`
# `TGDC ~ Weight : PIQ + MRI_Count`
# 
# We will chose `TGDC ~ Weight : Height + MRI_Count` as it has the best results

# In[12]:


super_model = 'TGDC ~ Weight : Height + MRI_Count'
super_model2 = 'TGCD ~ Weight : Height + MRI_Count'


# Fit the model
model= ols(super_model, data)

model_fit = model.fit()

# Print the summary
print(model_fit.summary())


# In[13]:


fig = plt.figure(figsize=(12,12))
fig = sm.graphics.plot_partregress_grid(model_fit, fig=fig)


# We can see that our data have some outliers. We will control find what z score to use for our dataset

# In[14]:


z_val = np.arange(1,3, 0.01)


inculded_vars = ['Height', 'Weight', 'MRI_Count']

df = pd.DataFrame(columns=['z_val', 'Rval', 'f_pvalue' ])


for val in z_val:
    
    z_s = np.abs(stats.zscore(data[inculded_vars]))
    dataf = data[(z_s < val).all(axis=1)]

    if dataf.shape[0] < 1:
        df = df.append( {'z_val':val, 'Rval':0, 'f_pvalue':0}, ignore_index=True)
        continue
    
    # Fit the model
    model= ols(super_model, dataf)

    rs = model.fit()
    
    df = df.append( {'z_val':val, 'Rval':rs.rsquared, 'f_pvalue':rs.f_pvalue}, ignore_index=True)
    
df.plot(x='z_val', y=['Rval', 'f_pvalue'], title='Z-score vs Rval')


# We can confirm that we don't need to remove any outliers in our sample. Another prove that our model really works

# In[15]:


# z_s = np.abs(stats.zscore(data[inculded_vars]))
# dataf = data[(z_s < 1.27).all(axis=1)]
# model= ols(super_model, dataf)

# rs = model.fit()

# print(rs.summary())
# fig = plt.figure(figsize=(12,12))
# fig = sm.graphics.plot_partregress_grid(rs, fig=fig)


# # Conculusion

# Lucky we measure the TGCD (Get from the Chair to the Door).
# 
# We expect that we can generalize the previous very acurate model to explain our data. 
# 
# To our suprize... does not give any significant results.
# 
# We may need to model in a different way this problem.

# In[16]:


# z_s = np.abs(stats.zscore(data[inculded_vars]))

# dataf = data[(z_s < 1.56).all(axis=1)]
# model= ols(super_model2, dataf)
model= ols(super_model2, data)

rs = model.fit()

print(rs.summary())
fig = plt.figure(figsize=(12,12))
fig = sm.graphics.plot_partregress_grid(rs, fig=fig)

