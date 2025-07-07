# Wood-models

This repository contains the code for a decision tree classifier, regressor, and random forest classifier machine learning model. 
The classification decision tree uses the Gini impurity to improve the tree, while the regression tree uses the variance.
A maximum depth, minimum samples per split and minimum samples per leaf can be given to control the tree.
Currently there is only a random forest model for classification problems.
In the future, other evaluation metrics and options like other score criteria could be added.
A random forest regressor could also be added.

The decision\_tree\_classifier and decision\_tree\_regressor directories contain a Jupyter notebook explaining how each tree works and a Python file with a class that contains the tree algorithm and could be imported into other scripts.
The random\_forest\_classifier directory similarly contains a Jupyter notebook explaining how the random forest works and a Python file with a class that contains the forest algorithm and could be imported into other scripts. 
Additionally, there is a modified decision tree algorithm which is needed for the random forest implementation.

Tests of both trees show that they generate predictions that are about as good as Sklearn's DecisionTreeClassifier/DecisionTreeRegressor, but they are slower.
Tests of the random forest similarly show that it generates predictions that are about as good as Sklearn's RandomForestClassifier, but with many trees it is much slower.
