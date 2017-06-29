
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[79]:


data=pd.read_csv('C:\\Users\\Naqsh\\Downloads\\asd\\diabetes.csv', names=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])


# In[80]:


data


# In[81]:


data["f9"].value_counts()


# In[82]:


sns.FacetGrid(data, hue="f9", size=5)    .map(plt.scatter, "f1", "f2")    .add_legend()


# In[83]:


#map data into arrays
s=np.asarray([1,0])
ve=np.asarray([0,1])
data['f9'] = data['f9'].map({'yes': s, 'no': ve})


# In[84]:


data


# In[85]:


#shuffle the data
data=data.iloc[np.random.permutation(len(data))]


# In[86]:


data


# In[87]:


data=data.reset_index(drop=True)


# In[88]:


data


# In[90]:


#training data
x_input=data.ix[0:450,['f1','f2','f3','f4','f5','f6','f7','f8']]
temp=data['f9']
y_input=temp[0:451]
#test data
x_test=data.ix[451:767,['f1','f2','f3','f4','f5','f6','f7','f8']]
y_test=temp[451:768]


# In[91]:


#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,8])
y_=tf.placeholder(tf.float32,shape=[None, 2])
#weight and bias
W=tf.Variable(tf.zeros([8,2]))
b=tf.Variable(tf.zeros([2]))


# In[92]:


# model 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[93]:


#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[94]:


#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[95]:


#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#number of interations
epoch=2000


# In[96]:


for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
    print (c)


# In[105]:


#random testing at Sn.130
a=data.ix[6,['f1','f2','f3','f4','f5','f6','f7','f8']]
b=a.reshape(1,8)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print ("Patient have Diabetes")
else :
    print ("Patient does not have Diabetes")

