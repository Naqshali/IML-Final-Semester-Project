IML FINAL PROJECT(On python Jupyter)

Classification, Newral Networks(Softmax Function) On the dataset to predict Whether Patients has diabetes or not.

The project is done on 2 datasets i.e. 1= Iris (Flower), 2= Patient Diabetes
There are 4 files proj1, proj2, proj3, proj4

Note: Iris-setosa, Iris-versicolor and Iris-virginica are the names of different kinds of iris flowers.

In Proj1:
Dataset is about Iris flower and we have to predict according to the features which is Iris-setosa, Iris-versicolor or Iris-virginica.


In Proj2:
Dataset is about Patient Diabetes and we have to predict according to the features which is patient have diabetes.

Steps and Algorithms For Proj1 and Proj 2 are same.

Steps:
1-First Load The dataset from the path.

2-On the next tab  it shows how many flowers are Iris-setosa, Iris-versicolor or Iris-virginica and or patient diabetes how many patients have diabetes or not.

3-Next tab shows the graphical outout according to the first 2 features only.It hows the classification for example if you run the iris dataset code you will see how Iris-setosa is calssified from Iris-versicolor and Iris-virginica.

5-Then, Outcome is converted into a matrix of ([0,0,0,] for iris dataset, [0,0] for Patient diabetes dataset).
 
6-The dataset for iris contains continous 15 rows for outcome Iris-setosa next 15 rows for outcome Iris-versicolor and then next 30 rows folowed by Iris-virginica.So, for training and test data we shuffled the dataset.  

7-Next tab shows the partition of dataset for training data and test data. 

8-Applying Machine Learning algo woth tensorfloe by placing tesorflow variable its weights and bias.

9- Next we are using SOFTMAX function in which we have to multiply our input features with weights and then addes to bias.

10-Finding  cross-entropy by default finction given which is actually to find loss function.

11-Next we are Optimizing the loss.

12- Variables and parameters initialization.

13-Loop for each iteration to fix loss.As we have to multiply row with vector so we are sending values as a matrix.

14-At end predicting the flower by giving a particular index.


In Proj3:
-Dataset: Patient Diabetes
-Predicting diabetes result by Logistic Regreesion.

Steps:
1-First Load The dataset from the path.

2-Showing Co-Realation Features.

3-Finding number of true and False Cases.

4-Next tab shows the partition of dataset for training data and test data and shows percentage of how much data is for training data and how much is for test data.

5-Next showing the percentage of training data and test data according to the true and false cases.

6-Finding Modle Accurasy by Gaussian algorithm from Naive Bayes.

7-Finding Modle Accurasy by Random Forest Algorithm.

8-Showing Accuraccy for training data.

9-Showing Accuraccy for training data.

10-Showing Classification Report.

11-At end showing accuracy by Logisti Regression Algorithm.

In Proj4:
-Dataset: Patient Diabetes
-Predicting diabetes result (accuraccy)by various algorihtms.

Steps:
1-First Load The dataset from the path.

2-Applying algorithm for 3 features for now.

3-Appending Algorithms.	

4-Showing Results
			Logistic Regression: 78.1229385307
			Naive Bayes: 77.6011994003
			K-Nearest Neighbour: 72.3883058471
			Decision Tree: 68.7556221889
			Support Vector Machine-linear: 77.9490254873
			Support Vector Machine-rbf: 65.7931034483
			Random Forest: 71.8770614693