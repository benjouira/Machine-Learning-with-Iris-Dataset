# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:31:46 2021

@author: rabi3
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn import neighbors
from ipywidgets import interact

iris = datasets.load_iris()


print(iris.feature_names)

print(iris.data[:5])

print(iris.data)
print(iris.target_names)

target = iris.target 
print(target)


for i in [0,1,2]:
    print("classe : %s, nb exemplaires: %s" % (i, len(target[ target == i]) ) )
    
    
print(iris.DESCR)


data = iris.data
print (type(data), data.ndim, data.shape)

#****** 
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax1 = plt.subplot(1,2,1)

clist = ['violet', 'yellow', 'blue']
colors = [clist[c] for c in iris.target]

ax1.scatter(data[:, 0], data[:, 1], c=colors)
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)')


ax2 = plt.subplot(1,2,2)
ax2.scatter(data[:, 2], data[:, 3], color=colors)
plt.xlabel('Longueur du petal (cm)')
plt.ylabel('Largueur du petal (cm)')

for ind, s in enumerate(iris.target_names):
    plt.scatter([], [], label=s, color=clist[ind])

plt.legend(scatterpoints=1, frameon=False, labelspacing=1
           , bbox_to_anchor=(1.8, .5) , loc="center right", title='Espèces')
plt.plot();

#**************
print (target)
sns.set()
df = pd.DataFrame(data, columns=iris['feature_names'] )
df['target'] = target
df['label'] = df.apply(lambda x: iris['target_names'][int(x.target)], axis=1)
print(df.head())

sns.pairplot(df, hue='label', vars=iris['feature_names'], size=2);

#*********Apprentissage
clf = GaussianNB()
clf.fit(data, target)
print(dir(clf))
print(clf.get_params())

result = clf.predict(data)
print(result)
print(result - target)

errors = sum(result != target) # 6 erreurs sur 150 mesures
print("Nb erreurs:", errors)
print( "Pourcentage de prédiction juste:", (150-errors)*100/150) 

#******calcule de pressision
print(accuracy_score(result, target))

#la matrice de confusion 
conf = confusion_matrix(target, result)
print(conf)


sns.heatmap(conf, square=True, annot=True, cbar=False
            , xticklabels=list(iris.target_names)
            , yticklabels=list(iris.target_names))
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles');
plt.matshow(conf, cmap='rainbow');

#split
data_test, target_test = data[::2], target[::2]
data_train, target_train = data[1::2], target[1::2]

print ('************',data_test, target_test, data_train, target_train)



# split the data with 50% in each set
data_test = train_test_split(data, target
                                 , random_state=0
                                 , train_size=0.5)
print ('************')
print (data_test)
data_train, data_test, target_train, target_test = data_test
print (data_test)
print(data_test[:5])

#****apprentissage 
clf = GaussianNB()
clf.fit(data_train, target_train)
result = clf.predict(data_test)

# Score
print(accuracy_score(result, target_test))

# Matrice de confusion
conf = confusion_matrix(target_test, result)
print(conf)

sns.heatmap(conf, square=True, annot=True, cbar=False
            , xticklabels=list(iris.target_names)
            , yticklabels=list(iris.target_names))
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles');

# On ne conserve que les longueurs/largeurs des sépales
print('**************************')
data = iris.data[:, :2]
target = iris.target
print(data[:5])


# On réapprend
clf = GaussianNB()
clf.fit(data, target)
h = .15
# Nous recherchons les valeurs min/max de longueurs/largeurs des sépales
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

x = np.arange(x_min, x_max, h)
y = np.arange(y_min, y_max, h)

print (x)
print (x_min, x_max, y_min, y_max)


xx, yy = np.meshgrid(x,y )
data_samples = list(zip(xx.ravel(), yy.ravel()) )
print('xx')
print(xx)
print('yy')
print(yy)


a = [ [10, 20],[ 1,  2] ]
print ('ravel')
print(np.array(a).ravel())

print('zip')
print(list(zip([10,20,30], [1,2])))

print(data_samples[:10])


Z = clf.predict(data_samples)
#Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z) 

# Plot also the training points
plt.scatter(data[:, 0], data[:, 1], c=target)
colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in Z]
plt.scatter(xx.ravel(), yy.ravel(), c=C)
plt.xlim(xx.min() - .1, xx.max() + .1)
plt.ylim(yy.min() - .1, yy.max() + .1)
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)');



plt.figure(1)
plt.pcolormesh(xx, yy, Z.reshape(xx.shape)) # Affiche les déductions en couleurs pour les couples x,y
# Plot also the training points
colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in target]
plt.scatter(data[:, 0], data[:, 1], c=C)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)');



clf = neighbors.KNeighborsClassifier()

@interact(n=(0,20))
def n_change(n=5):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    clf.fit(data, target)
    Z = clf.predict(data_samples)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z.reshape(xx.shape)) # Affiche les déductions en couleurs pour les couples x,y
    # Plot also the training points
    colors = ['violet', 'yellow', 'red']
    C = [colors[x] for x in target]
    plt.scatter(data[:, 0], data[:, 1], c=C)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Longueur du sepal (cm)')
    plt.ylabel('Largueur du sepal (cm)');


data_test, target_test = iris.data[::2], iris.target[::2]
data_train, target_train = iris.data[1::2], iris.target[1::2]
result = []
n_values = range(1,20)
for n in n_values:
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    clf.fit(data_train, target_train)
    Z = clf.predict(data_test)
    score = accuracy_score(Z, target_test)
    result.append(score)

plt.plot(list(n_values), result)

print("**********Apprentissage non supervisé***************")

from sklearn.decomposition import PCA
# Définition de l'hyperparamètre du nombre de composantes voulues
model = PCA(n_components=2)
# Alimentation du modèle
model.fit(iris.data)
# Transformation avec ses propres données
reduc = model.transform(iris.data )

print(iris.data[:5])
print(reduc[:5])
print(df.head())

df['PCA1'] = reduc[:, 0]
df['PCA2'] = reduc[:, 1]
print(df.head())

colors = ['violet', 'yellow', 'blue']
plt.scatter(df['PCA1'], df['PCA2'], c=[ colors[c] for c in df['target'] ]);
plt.xlabel('PCA1')
plt.ylabel('PCA2');
sns.lmplot("PCA1", "PCA2", hue='label', data=df, fit_reg=False);


from sklearn.mixture import GaussianMixture
# Création du modèle avec 3 groupes de données
model = GaussianMixture (n_components=3, covariance_type='full')
# Apprentissage, il n'y en a pas vraiment
model.fit(df[['PCA1', 'PCA2']])
# Prédiction
groups = model.predict(df[['PCA1', 'PCA2']])

df['group'] = groups
print(df.head(5))
sns.lmplot("PCA1", "PCA2", data=df, hue='label',
           col='group', fit_reg=False);
