# MachineLearning_Projects
This repository consists of various Machine Leraning related projects as part of my course curriculum at University of Cincinnati.
Each one of them has its own data sets and are solve different problems. 
Both supervised and unsupervised machine learning algorithms are implemented.

## Badges_Game.py
This project implements a supervised machine learning algorithm(Decision Tree) to solve a classic ML problem called the **Badges Problem**.
**"Information Gain(Entropy)"** criterion is used in designing the decision tree model
An external too called **"dot tool"** is used to print the decision tree designed in the project.

## Binning.py
This is a basic statistical analysis project where the data rows are binned using both equal frequency and equal width partitioning techniques.
Then to further achieve accurate results **z-score normalization** is implemented and then the binning is performed again with the normalized values and the difference have been documented.

## InformationGain_Calculator.py
Based on the z-score values calculated in the earlier module, labels have been assigned for two of the attributes and the attribute best suited to predict the predictor variable is identified by calculating the **Information Gain** value individually for either of them.

##DecisionTree.py
The supervised decision tree algorithm is implemented for the **Magic Gamma Telescope Dataset**. (https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope) consisting of around **20000** records.
The data was divided randomly into **Training**, **Validation** and **Testing** partitions.
Training data was used to deisgn and train the decision tree classifier using **Information Gain(entropy)** criterion. Accuracy of **81%** was achieved. Then the model was validated using the validation data.
To prevent the model from overfitting, mutiple classifiers were designed with varying number of leaf nodes.
All these models were tested against the testing data and the best model(differed on the number of leaf nodes) was identified by comparing accuracy, precision and recall values.
ROC curve was plotted using **Matplotlib** library and also **Dot tool** was used to print the classifier.
*It was observed that the decision tree model having minimum of 20 nodes at the leaf level was the optimal one with an accuracy of **84%**.*

##Classifier_Comparator.py
This project makes use of a well known dataset called **Wisconsin Breast-Cancer Dataset**(http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29 ) to design and train multiple classifiers and choose the optimal one based on the metric values identified.
The dataset again is partitioned into training and testing subsets.
Different Machine learning algorithms like **Decision Trees** , **Support Vector Machines**, **k-Nearest Neighbors** were used to train multiple models based on the training data.
The **SVM** model was identified as the best for this data with an accuracy of **98%**, precision of **99.2%** and recall value of **97.6%**.
The **Misclassification** costs were also calculated for each of the different models to select the best and precise model.

##Clustering.py
In this project I was able to implement, observe and compare different un-supervised machine learning algorithms like **Basic Sequential Clustering(BSAC)**, multiple variants of **Hierarchical clustering** and also **DBSCAN**.
Random data points were generated to implement the basic clustering algorithm with a fixed **theta(threshold)** value. The order of the data points was later flipped to observe the dependency on the ordering of the points. **Adjusted RAND index** metric was calculated to find the difference between the two clusterings and for the data used, a significant value of 14% was identified.
For the same data **Hierarchical clustering** was performed using both **single-link** and **complete-link** distance measures and both the **dendrograms** was printed and analysed.
In each of the clustering the **Sum of Squared Error(SSE)** was calculated and the cluster with the highest **Sum of Squared Error(SSE)** was identified.
**Cophenetic Correlation(a measure to determine how a dendrogram preserves pair wise distances between unmodelled data points) value was computed for both the clusterings. From the values obtained it was observed that the **complete-link** method provided more correlated results for the data used
Also to perform cluster validation both **proximity** and **incidence** matrices were created for both the distance measures and **correlation coefficients** were computed. Again it was observed that the **complete-link** scenario had a higher correlation coefficient value(0.741).
Finally a set of 1-D points were used to implement **DBSCAN** algorithm with multiple **epsilon** and **MinPoints** parameter values.
The labels assigned to the points were printed and analysed and it was observed that as the **epsilon** value increases more and more points are labelled as **core** instead of being **border** points. Also the **adjusted RAND index** value showed that both the clustering were very different(RAND index = 0.087) even though the difference in the **epsilon** value is ver low.
These results showed that the **DBSCAN** algorithm is mostly **resistant** to noise but is highly dependent on the **epsilon** value selected.
