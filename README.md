# Wood-models

This repository contains the code for a decision tree classifier machine learning model. 
Currently, it only deals with classification problems, uses the Gini impurity to improve the tree, and a maximum depth, minimum samples per split and minimum samples per leaf can be given to control the tree.
In the future, other evaluation metrics and options like other score criteria could be added.
Additionally, an algorithm to deal with regression problems could be added.

The decision\_tree\_classifier directory contains a Jupyter notebook explaining how the tree works and a Python file with a class that contains the tree algorithm and could be imported into other scripts.
The random\_forest\_classifier directory similarly contains a Jupyter notebook explaining how the random forest works and a Python file with a class that contains the forest algorithm and could be imported into other scripts. 
Additionally, there is a modified decision tree algorithm which is needed for the random forest implementation.

Tests of the tree show that it generates predictions that are about as good as Sklearn's DecisionTreeClassifier, but it is slower.
Tests of the random forest similarly show that it generates predictions that are about as good as Sklearn's RandomForestClassifier, but with many trees it is much slower.
