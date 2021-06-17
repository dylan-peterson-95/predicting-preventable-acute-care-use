#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Generating Simulated Data

# In[2]:


# read in our real data
df = pd.read_csv("../data/feature_matrices/FULL_KNN.csv")

# generate an empty data frame to hold 100 simulated patients
sim_df = pd.DataFrame(columns=df.columns, index=range(100))


# In[3]:


# find our binary columns
binary_cols = df.loc[:, df.isin([0, 1]).all()].columns

# find our integer columns
int_cols = [(df[col] % 1 == 0).all() for col in df.columns]
int_cols = df.columns[int_cols]


# In[4]:


# iterate through our data fields
for col in df.columns:

    # if binary, draw from a random bernoulli
    if col in binary_cols:
        sim_df[col] = np.random.binomial(n=1, p=df[col].mean(), size=100)

    # otherwise, draw from a random gaussian, 
    # but take the absolute value as none of our data should be negative
    else:
        sim_df[col] = np.round(abs(
            np.random.normal(loc=df[col].mean(), scale=df[col].std(),
                             size=100)),
                               decimals=2)

    # finally, if a count value, transform to an integer
    if col in int_cols:
        sim_df[col] = sim_df[col].astype(int)


# In[5]:


# save the results
sim_df.to_csv("../data/simulated_data/simulated_feature_matrix.csv",
              index=False)
sim_df.head()


# In[6]:


# generate simulated outcomes p=0.35 comes from the incidence of OP-35 events at 30 days in our cohort
sim_outcomes = pd.DataFrame(sim_df["PAT_DEID"])
sim_outcomes["ANY_180"] = np.random.binomial(n=1, p=0.35, size=100)

# save the results
sim_outcomes.to_csv("../data/simulated_data/simulated_outcomes.csv",
              index=False)
sim_outcomes.head()

