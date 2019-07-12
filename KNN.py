# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:19:02 2019

@author: s16154368
"""

import os
import pandas as pd 
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


df2 = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
ax = df2.plot.bar(x='lab', y='val', rot=0)
zoo = pd.read_csv('Z:/AI/zoo.csv')
print(zoo)

zoo=zoo.drop('Animal_Name',axis=1)

X=zoo.iloc[:,:16].values
print(X)
y=zoo.iloc[:,16].values
print(y)

zoo.shape
parameters = {'n_neighbors':[4,5,6,7],
          'leaf_size':[1,3,5],
           'algorithm':['auto', 'kd_tree'],
            'n_jobs':[-1]}


ax = zoo['Airborne'].value_counts().plot(kind = 'bar', title = "Instances of Airborne Animals")
ax.set_xlabel("Airborne or Not Airborne")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Airborne","Not Airborne"])
plt.show()

ax = zoo['Aquatic'].value_counts().plot(kind = 'bar', title = "Instances of Aquatic Animals")
ax.set_xlabel("Aquatic or Not Aquatic")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Aquatic","Not Aquatic"])
plt.show()

ax = zoo['Predator'].value_counts().plot(kind = 'bar', title = "Instances of Predators")
ax.set_xlabel("Predator or No Predator")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Predator","No Predator"])
plt.show()

ax = zoo['Backbone'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Backbone")
ax.set_xlabel("Backbone or No backbone")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Backbone","No Backbone"])
plt.show()

ax = zoo['Toothed'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Teeth")
ax.set_xlabel("Backbone or No backbone")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Toothed","No Toothed"])
plt.show()

ax = zoo['Breathes'].value_counts().plot(kind = 'bar', title = "Instances of Animals which Resperates")
ax.set_xlabel("Breathes or Does Not Breathe")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Breathes","Does not Breathe"])
plt.show()

ax = zoo['Venomous'].value_counts().plot(kind = 'bar', title = "Instances of Venoumous Animals")
ax.set_xlabel("Venomous or Not Venomous")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Venomous","Not Venomous"])
plt.show()

ax = zoo['Fins'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Fins")
ax.set_xlabel("Fins or No Fins")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Fins","No Fins"])
plt.show()

ax = zoo['Hair'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Hair")
ax.set_xlabel("Hair or No Hair")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Hair","No Hair"])
plt.show()

ax = zoo['Eggs'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Eggs")
ax.set_xlabel("Eggs or No Eggs")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Eggs","No Eggs"])
plt.show()

ax = zoo['Milk'].value_counts().plot(kind = 'bar', title = "Instances of Animals with Milk")
ax.set_xlabel("Milk or No Milk")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Milk","No Milk"])
plt.show()

ax = zoo['Fins'].value_counts().plot(kind = 'bar', title = "Instances of Fins")
ax.set_xlabel("Fins or No Fins")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Fins","No Fins"])
plt.show()

ax = zoo['Legs'].value_counts().plot(kind = 'bar', title = "Number of legs ")
ax.set_xlabel("Number of Legs")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Legs","No Legs"])
plt.show()

ax = zoo['Tail'].value_counts().plot(kind = 'bar', title = "Number of Animals with Tails")
ax.set_xlabel("Tail or No Tail")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Tail","No Tail"])
plt.show()

ax = zoo['Domestic'].value_counts().plot(kind = 'bar', title = "Number of Domestic Animals")
ax.set_xlabel("Domestic or No Domestic")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Domestic","No Domestic"])
plt.show()

ax = zoo['Catsize'].value_counts().plot(kind = 'bar', title = "Instances of Catsize")
ax.set_xlabel("Catsize or No Catsize")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Catsize","No Catsize"])
plt.show()
 
ax = zoo['Type'].value_counts().plot(kind = 'bar', title = "Instances of Types")
ax.set_xlabel("Types")
ax.set_ylabel("Number of Instances")
ax.xaxis.set(ticklabels=["Type1","Type2","Type 3","Type 4","Type 5","Type 6","Type 7"])
plt.show()


# Preprocesing - Inputted in the console
zoo.Hair.describe()
zoo.Feathers.describe()
zoo.Eggs.describe()
zoo.Milk.describe()
zoo.Airborne.describe()
zoo.Aquatic.describe()
zoo.Predator.describe()
zoo.Toothed.describe()
zoo.Backbone.describe()
zoo.Breathes.describe()
zoo.Venomous.describe()
zoo.Fins.describe()
zoo.Legs.describe()
zoo.Tail.describe()
zoo.Domestic.describe()
zoo.Catsize.describe()
zoo.Type.describe()



#This is for the decision tree classifier splitting the dataset for this algorithm
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state= 50)

#KNeighbours CLASSIFER
clf =KNeighborsClassifier() 
clf.fit (x_train, y_train)
score = clf.score(x_test, y_test)
y_pred=clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


#DEFINE PARAMETERS FOR HIGHEST SCORE FOR KNeighbors CLASSIFIER  
param_grid = {'n_neighbors': [5, 2]}

#GRID SEARCH FOR DECISION TREE CLASSIFIER
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
start = time()

grid_search.fit(x_train, y_train)
y_pred=grid_search.predict(x_test)
#print(grid_score=grid_search.score(y_test, y_pred))

#printing results
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_)


#CROSS VALIDATION

scores = cross_val_score(clf, X, y, cv=10)
mean_tree_score = scores.mean()
print("Mean Accuracy: %0.2f" % (scores.mean()))



#CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["1","2","3","4","5","6","7"],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["1","2","3","4","5","6","7"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()



#GRAPH OF ALL THE ACCURACY SCORES
x = [u'DecsisonTreeClassifier', u'NeuralNetworks', u'SVM', u'NearestNeighbour']
y = [0.94, 1.0, 0.83, 0.93]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Algorithm comparison')
plt.xlabel('x')
plt.ylabel('y')      
plt.show()
plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial picture

