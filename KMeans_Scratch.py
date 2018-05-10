import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

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

def Centroid_init(K):

	centroid=[]
	for i in range(K):
		centroid.append(random.choice(X))
	centroid=np.array(centroid)
	return centroid

def Labeling(centroid):
	label=[]
	for i in range(len(X)):
		A = X[i]
		distance=[]
		for e in range(K):
			B = centroid[e]
			# distance.append(np.sqrt(np.sum(A-B)**2))
			D=np.sqrt(np.sum(A-B)**2)
			# print(A ,B)
			# print(D)
			distance.append(D)
		idx = np.argmin(distance)
		# print(idx, distance[idx])
		# print('\n')
		label.append(idx)
	# print(label)
	return label

def Centroid_change(label, K):

	new_cent=[]
	for i in range(K):
		new_cent_i=[]
		for e in range(len(label)):
			if label[e] == i:
				nc = label[e]
				new_cent_i.append(X[e])
			else :
				pass
		new_cent_i=np.array(new_cent_i)
		# var_test=new_cent_i.sum(axis=0)/len(new_cent_i)
		new_cent.append(new_cent_i.sum(axis=0)/len(new_cent_i))

	new_cent=np.array(new_cent)
	# print(new_cent)
	return new_cent

def Accuracy(Final_label, Y):

	count=0
	for i in range(len(Y)):
		if Y[i]==Final_label[i]:
			count=count+1
		else:
			pass
	accuracy=count/len(Y)

	return accuracy, count

def KMeans():

	Initial_cen=Centroid_init(K)
	Cent_Store=[]
	count=0
	while True:
		if count == 0:
			Label_labeling = Labeling(Initial_cen)
		else :
			Label_labeling = Labeling(Changed_centroid)

		Changed_centroid = Centroid_change(Label_labeling, K)
		# print(Changed_centroid)
		Cent_Store.append(Changed_centroid)
		if count == 0:
			pass
		else:	
			count_p=count-1
			if Cent_Store[count_p].all()==Cent_Store[count].all():
				break
			else:
				pass
		count=count+1
		# print(count)

	Final_labeling=Labeling(Changed_centroid)
	Final_labeling=np.array(Final_labeling)
	return Final_labeling, Changed_centroid, count

K=3
itt=30

Final_label_itt=[]
Final_Centroid_itt=[]
Itteration_itt=[]
Accuracy_itt=[]
Count_iit=[]

for i in range(itt):
	Final_label, Final_Centroid, Itteration=KMeans()
	# print(Y)
	# print(Final_label)
	# print(Final_Centroid)
	# print(Itteration)
	Acc, cou =Accuracy(Final_label, Y)
	Final_label_itt.append(Final_label)
	Final_Centroid_itt.append(Final_Centroid)
	Itteration_itt.append(Itteration)
	Accuracy_itt.append(Acc)
	Count_iit.append(cou)

# print(Accuracy_itt)

Max_id=np.argmax(Accuracy_itt)
# print(Accuracy_itt)
# print(Max_id)
print(Accuracy_itt[Max_id])
# print(Count_iit[Max_id])

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
