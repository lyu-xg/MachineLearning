from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

# load and print data
iris = load_iris()
for i in range(len(iris.target)):
    print 'Data Entry {:3}: labal: {:10}, features {}'.format(i,iris.target_names[iris.target[i]],iris.data[i])

# remove test dataset from imported dataset
test_idx = [1,51,101] # happens to be the second entry for each label
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print 'prediction', clf.predict(test_data)
print 'actual labels', test_target

# visualization
with open("iris.dot", 'w') as f:
     f = tree.export_graphviz(clf, out_file=f)

import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
from IPython.display import Image

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
