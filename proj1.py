
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[267]:


data=pd.read_csv('C:\\Users\\Naqsh\\Downloads\\asd\\iris.csv', names=['f1','f2','f3','f4','f5'])


# In[269]:


data


# In[270]:


data["f5"].value_counts()


# In[271]:


sns.FacetGrid(data, hue="f5", size=5)    .map(plt.scatter, "f1", "f2")    .add_legend()


# In[272]:


#map data into arrays
s=np.asarray([1,0,0])
ve=np.asarray([0,1,0])
vi=np.asarray([0,0,1])
data['f5'] = data['f5'].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})


# In[273]:


data


# In[274]:



#shuffle the data
data=data.iloc[np.random.permutation(len(data))]


# In[275]:


data


# In[276]:


data=data.reset_index(drop=True)


# In[277]:


data


# In[281]:


#training data
x_input=data.ix[0:30,['f1','f2','f3','f4']]
temp=data['f5']
y_input=temp[0:31]
#test data
x_test=data.ix[31:58,['f1','f2','f3','f4']]
y_test=temp[31:59]


# In[282]:


#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])
#weight and bias
W=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))


# In[283]:


# model 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[284]:


#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[285]:


#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[286]:


#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#number of interations
epoch=2000


# In[287]:


for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
    print (c)


# In[291]:


#random testing at Sn.130
a=data.ix[8,['f1','f2','f3','f4']]
b=a.reshape(1,4)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print ("flower is :Iris-setosa")
elif largest==1:
    print ("flower is :Iris-versicolor")
else :
    print ("flower is :Iris-virginica")


# In[292]:


print sess.run(accuracy,feed_dict={x: x_test, y_:[t for t in y_test.as_matrix()]})


# In[ ]:




