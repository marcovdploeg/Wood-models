# Wood-models

This repository contains the code for a decision tree classifier machine learning model. 
Currently, it only deals with classification problems, uses the Gini impurity to improve the tree, and a maximum depth can be given at which it stops building the tree.
In the future, other evaluation metrics and options like a minimum amount of samples per leaf could be added.
Additionally, an algorithm to deal with regression problems could be added.
I also plan to expand this into a random forest model, using this tree algorithm as its basis.

The decision_tree directory contains a Jupyter notebook explaining how the tree works and a Python file with a class that contains the tree algorithm and could be imported into other scripts.

Tests of the tree show that it generates predictions that are about as good as sklearn's DecisionTreeClassifier, but it is slower.
