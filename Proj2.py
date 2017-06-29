
# coding: utf-8

# In[46]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[47]:


data=pd.read_csv('C:\\Users\\Naqsh\\Downloads\\asd\\diabetes.csv', names=['Pragnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])


# In[48]:


data


# In[49]:


data["Outcome"].value_counts()


# In[50]:


sns.FacetGrid(data, hue="Outcome", size=5)    .map(plt.scatter, "Pragnancies", "Glucose")    .add_legend()


# In[51]:


sns.FacetGrid(data, hue="Outcome", size=5)    .map(plt.scatter, "Blood Pressure","Skin Thickness")    .add_legend()


# In[52]:


sns.FacetGrid(data, hue="Outcome", size=5)    .map(plt.scatter, "Insulin","BMI")    .add_legend()


# In[53]:


sns.FacetGrid(data, hue="Outcome", size=5)    .map(plt.scatter, "DiabetesPedigreeFunction","Age")    .add_legend()


# In[54]:


#map data into arrays
s=np.asarray([1,0])
ve=np.asarray([0,1])
data['Outcome'] = data['Outcome'].map({'yes': s, 'no': ve})


# In[55]:


data


# In[56]:


#shuffle the data
data=data.iloc[np.random.permutation(len(data))]


# In[57]:


data


# In[58]:


data=data.reset_index(drop=True)


# In[59]:


data


# In[64]:


#training data
x_input=data.ix[0:450,['Pragnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
temp=data['Outcome']
y_input=temp[0:451]
#test data
x_test=data.ix[451:767,['Pragnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y_test=temp[451:768]


# In[38]:


#placeholders and variables. input has 8 features and output has 2 classes
x=tf.placeholder(tf.float32,shape=[None,8])
y_=tf.placeholder(tf.float32,shape=[None, 2])
#weight and bias
W=tf.Variable(tf.zeros([8,2]))
b=tf.Variable(tf.zeros([2]))


# In[39]:


# model 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[40]:


#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[41]:


#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[42]:


#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#number of interations
epoch=2000


# In[43]:


for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
    print (c)


# In[45]:


#random testing at Sn.130
a=data.ix[6,['Pragnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
b=a.reshape(1,8)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print ("Patient have Diabetes")
else :
    print ("Patient does not have Diabetes")


# In[ ]:




