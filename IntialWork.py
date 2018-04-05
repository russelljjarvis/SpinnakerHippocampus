
# coding: utf-8

# In[4]:


get_ipython().system('pip install elephant')


# In[1]:


import neo 
import elephant


# In[2]:


help(elephant)


# In[3]:


import pickle
mdbf = pickle.load( open("Documents/NeuroProject/pickles/membrane_dynamics_balanced_file.p", "rb" ) )


# In[4]:


dir(mdbf)
ass = mdbf.analogsignals
spike_trains = mdbf.spiketrains


# In[5]:


len(ass)


# In[6]:


dir(ass)


# In[40]:


#dir(spike_trains)
#len(spike_trains)
from elephant.statistics import cv
import matplotlib.pyplot as plt

hist_cv = []
import numpy as np
for i in spike_trains:
    cva = cv(i)
    if np.isnan(cva):
        hist_cv.append(0)
    else:    
        hist_cv.append(cva)
   # print(cv(i))
x_axis = [i for i in range(0,len(hist_cv))]
plt.bar(x_axis,hist_cv)
plt.show()
plt.hist(hist_cv)




#dir(elephant)
#plt.


# In[46]:


unested_sp = []
for s in spike_trains:
    unested_sp.extend(s)
isi_hist = elephant.statistics.isi(unested_sp)
#print(isi_hist)


# In[38]:


# Taking a look at the distribution without zeros
#plt.hist(hist_cv.
hist_cv = [i for i in hist_cv if i!=0 ]
print(hist_cv)
plt.hist(hist_cv,bins=100)
print(isi_hist)

#plt.show()

#plt.show()


# In[51]:


#elephant.statistics.mean_firing_rate(spike_trains)
np.max(spike_trains)


# In[69]:




