# ==================================== Importing Libraries ================================================

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# ===================================== Reading data file =================================================

iris=pd.read_csv("iris.data")
df = pd.DataFrame(iris)

Y=np.array(iris)
Y=np.delete(Y,np.s_[0:4],1)
for i in range(len(Y)):
	if Y[i] == 'Iris-setosa':
		Y[i]=0
	elif Y[i] == 'Iris-versicolor':
		Y[i]=1
	elif Y[i] == 'Iris-virginica':
		Y[i]=2
	else:
		pass
Y=np.reshape(Y,len(Y))
X=np.array(iris)
X=np.delete(X,4,1)

# ===================================== Defining Functions =================================================

#Random centroid initiation
def Centroid_init(K):

	centroid=[]
	for i in range(K):
		centroid.append(random.choice(X))
	centroid=np.array(centroid)

	return centroid

#Labeling data based on centroid
def Labeling(centroid):

	label=[]
	label_bor=[]
	for i in range(len(X)):
		A = X[i]
		distance=[]
		for e in range(len(centroid)):
			B = centroid[e]
			D=np.sqrt(np.sum(A-B)**2)
			distance.append(D)
		idx = np.argmin(distance)
		idx_max = np.argmax(distance)
		label.append(idx)
		label_bor.append(idx_max)

	return label, label_bor

#New centroid from labeled data i.e, Avg of labeled.pt
def Centroid_change(label, K_cent):

	new_cent=[]
	for i in range(K_cent):
		new_cent_i=[]
		for e in range(len(label)):
			if label[e] == i:
				nc = label[e]
				new_cent_i.append(X[e])
			else :
				pass
		new_cent_i=np.array(new_cent_i)
		if len(new_cent_i)==0:
			pass
		else:
			new_cent.append(new_cent_i.sum(axis=0)/len(new_cent_i))
	new_cent=np.array(new_cent)

	return new_cent

#Accurace function 
def Accuracy(Final_label, Y):

	count=0
	for i in range(len(Y)):
		if Y[i]==Final_label[i]:
			count=count+1
		else:
			pass
	accuracy=count/len(Y)

	return accuracy, count

#KMeans function which itterates untill there is no change in centroid
def KMeans():

	Initial_cen=Centroid_init(K)
	Cent_Store=[]
	count=0
	while True:
		if count == 0:
			Label_labeling, _ = Labeling(Initial_cen)
		else :
			Label_labeling, _ = Labeling(Changed_centroid)
		Changed_centroid = Centroid_change(Label_labeling, K)
		Cent_Store.append(Changed_centroid)
		if count == 0:
			pass
		else:	
			count_p=count-1
			if np.all(Cent_Store[count_p]==Cent_Store[count]):
				break
			else:
				pass
		count=count+1
	Final_labeling, Final_labeling_bor=Labeling(Changed_centroid)
	Final_labeling=np.array(Final_labeling)

	return Final_labeling, Final_labeling_bor, Changed_centroid, count

# ==================================== Initial KMeans operation =========================================

K=3
itt=30
Final_label_itt=[]
Final_label_bor_itt=[]
Final_Centroid_itt=[]
Itteration_itt=[]
Accuracy_itt=[]
Count_iit=[]

#Itterates untill reaches global optima as KMeans may converges at local optima based on initial centroid
for i in range(itt):
	Final_label, Final_label_bor, Final_Centroid, Itteration=KMeans()
	Acc, cou =Accuracy(Final_label, Y)
	Final_label_itt.append(Final_label)
	Final_label_bor_itt.append(Final_label_bor)
	Final_Centroid_itt.append(Final_Centroid)
	Itteration_itt.append(Itteration)
	Accuracy_itt.append(Acc)
	Count_iit.append(cou)
Max_id=np.argmax(Accuracy_itt)
#Accuracy
print(Accuracy_itt[Max_id])
#Count of failed predictions
print(len(X)-Count_iit[Max_id])

# ========================================= Visualization ==============================================

plt.scatter(Final_Centroid_itt[Max_id][0][0],Final_Centroid_itt[Max_id][0][3],marker='x',c='r')
plt.scatter(Final_Centroid_itt[Max_id][1][0],Final_Centroid_itt[Max_id][1][3],marker='x',c='g')
plt.scatter(Final_Centroid_itt[Max_id][2][0],Final_Centroid_itt[Max_id][2][3],marker='x',c='b')
# plt.scatter(X[:,0],X[:,1])
for i in range(len(X)):
	if Final_label_itt[Max_id][i]==0:
		plt.scatter(X[i][0],X[i][3], s=5, c='r')
	elif Final_label_itt[Max_id][i]==1:
		plt.scatter(X[i][0],X[i][3], s=5, c='g')
	elif Final_label_itt[Max_id][i]==2:
		plt.scatter(X[i][0],X[i][3], s=5, c='b')
	else:
		pass		
plt.show()