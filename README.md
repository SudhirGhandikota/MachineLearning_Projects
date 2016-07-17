# MachineLearning_Projects
This repository consists of various Machine Leraning related projects as part of my course curriculum at University of Cincinnati.
Each one of them has its own data sets and are solve different problems. 
Both supervised and unsupervised machine learning algorithms are implemented.

## Badges_Game.py
This project implements a supervised machine learning algorithm(Decision Tree) to solve a classic ML problem called the "Badges Problem".
"Information Gain" measure is used in designing the decision tree model
An external too called "dot tool" is used to print the decision tree designed in the project.

## Binning.py
This is a basic statistical analysis project where the data rows are binned using both equal frequency and equal width partitioning techniques.
Then to further achieve accurate results z-score normalization is implemented and then the binning is performed again with the normalized values and the difference have been documented.
Also based on the z-core values, labels have been assigned for two of the attributes and attribute most suited to predict the predictor variable is identified by calculating the "Information Gain" individually for either of them.

##DecisionTree.py
The supervised decision tree algorithm is implemented for the "Magic Gamma Telescope Dataset" (https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope) consisting of around 20000 records.
The data was divided randomly into Training, Validation and Testing partitions.
Training data was used to deisgn and train the decision tree classifier using "Information Gain(entropy)" criterion. Accuracy of 81% was achieved. Then the model was validated using the validation data.
To prevent the model from overfitting, mutiple classifiers were designed with varying number of leaf nodes.
All these models were tested against the testing data and the best model(differed on the number of leaf nodes) was identified by comparing accuracy, precision and recall values.
*It was observed that the decision tree model having minimum of 20 nodes at the leaf level was the optimal one*
