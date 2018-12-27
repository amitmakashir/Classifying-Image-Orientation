#!/usr/bin/env python3

'''
https://machinelearningmastery.com/implement-random-forest-scratch-python/
https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

As the name suggests, a Random forest is a collection of Decision trees. It is bagging approach
where multiple Decision trees are constructed with different features and subset of training data.
To better understand how the Random forest algorithm works we must first understand how a
Decision tree works.

A Decision tree creates a tree like structure with Decision nodes and leaf(s). A Decision node is
rule or condition for splitting the data into two buckets, one where the condition is true and other
where it’s not. This split is done on one value of one feature, where the Information gain is maximum
(or in other words this split produces least impure buckets of data). To measure this impurity
we have Gini impurity. If you are fortunate, after some splits you’ll get to a point where one
of your bucket has data points of only one class. We’ll call this a Leaf node. We have implemented
two conditions to stop growing the tree, a Leaf node and maximum depth of tree.

In a Random forest, we build “N” such Decision trees and take their vote. So you can tune the
Random forest by changing the number of trees it builds, the percent of data used to build each
tree, no. of features used to split and maximum depth of tree. As you increase the number of trees
your variance decreases but the training time also increases. As you increase the percent of data
used to build each tree, your trees start becoming similar to each other and give a similar and
probably better prediction but also increases training time. As you increase no. of features and
maximum depth of tree you your predictions get better up to certain value and then decrease from there.

Initially, we considered every possible split to get the best split at each node. This means,
if you have “f” features and each of those feature has 255 unique values, then you would be checking
f*255 splits. To overcome this problem, we sort the unique values of a feature and split only at
points that have a cluster of same class data points. This helps reduce split points and thus
significantly reduces training time.


'''
import numpy as np
import sys
import os
import time
import math
import pickle


os.chdir(os.getcwd())
from machinelearning import file_load


class DecisionTree:
    '''
    depth = Depth of the Decision tree
    max_features = ref:sklearn
    min_sample_leaf = ref:sklearn
    '''

    def __init__(self, depth,max_features):
        self.depth = depth
        self.max_features = max_features


    def fit(self, X, y):
        y_classes = self.purity_check(y)

        # If the info_gain is 0 for the first split, we'll get leaf here
        # This is very unlikely though

        # If the y array only has 1 class in it, just make the root node as Leaf
        if len(y_classes.keys()) == 1:
            self.root_node = Leaf(y)
        else:
            # Calculate all unique values for every column and store it as a class variable
            self.get_random_features(X)
            self.unique_val = self.get_unique_val(X)
            self.root_node = self.build_tree(X, y, 0)


    def build_tree(self, X, y, depth):
        # Find the best split point that gives the highest information gain
        # for eg: split = {"col_index":1,"value":3}
        info_gain, split = self.get_split_point(X, y)

        if info_gain == 0 or depth >= self.depth:
            return Leaf(y)

        true_X, true_y, false_X, false_y = self.segregate(X, y, split)

        # Now that you have rows that are split in true and false, repeat the same procedure again recursively
        # till you get a split where the info_gain is 0 or other stopping condition (like depth, min_leaf_node)
        depth += 1
        # print("depth:"+str(depth))
        true_side = self.build_tree(true_X, true_y, depth)
        false_side = self.build_tree(false_X, false_y, depth)

        '''
        Every Decision node will have 3 elements:
        1. split: the feature and its value to split on
        2. true_side: Can have rows or a leaf node. rows are the ones that satisfied the split condition
        3. false_side: Can have rows or a leaf node. rows are the ones that didn't satisfiy the split condition
        '''
        return Decision_Node(split, true_side, false_side)


    def segregate(self, X, y, split):
        col = split["col_index"]
        val = split["value"]

        true_indices = np.where(X[:,col] >= val)
        true_X = X[true_indices]
        true_y = y[true_indices]

        false_indices = np.where(X[:,col] < val)
        false_X = X[false_indices]
        false_y = y[false_indices]

        return true_X, true_y, false_X, false_y


    def get_split_point(self, X, y):

        # Find the current Gini index
        current_gini = self.gini_impurity(y)

        max_info_gain = 0
        optimal_col_index = 0
        optimal_value = 0

        # Loop through all columns to get the best split
        for col in self.sampled_features:
            # Find unique values in each column and split it
            values_to_split = self.split_values(X[:,col],y)

            for v in values_to_split:

                split = {"col_index": col, "value": v}
                true_X, true_y, false_X, false_y = self.segregate(X, y, split)

                # Find Information gain by this split
                ig,pure_split = self.info_gain(current_gini, true_y, false_y)

                if ig > max_info_gain:
                    max_info_gain = ig
                    optimal_col_index = col
                    optimal_value = v

                # No need to run the entire loop if you get a pure split
                if pure_split:
                    info_gain = max_info_gain
                    split = {"col_index": optimal_col_index, "value": optimal_value}
                    return info_gain, split

        info_gain = max_info_gain
        split = {"col_index": optimal_col_index, "value": optimal_value}
        return info_gain, split


    def gini_impurity(self, y):
        class_distribution = self.purity_check(y)
        # impurity = 1
        # for class_label in class_distribution:
        #     prob_label = class_distribution[class_label] / float(len(y))
        #     impurity -= prob_label ** 2
        total = float(len(y))
        # impurity = sum([(class_distribution[i]/total)*(1-(class_distribution[i]/total)) for i in class_distribution])
        prob = np.array([class_distribution[i] / total for i in class_distribution.keys()])
        impurity = 1 - np.sum(np.square(prob))
        return impurity


    def purity_check(self, y):
        class_distribution = {}
        for class_label in y:
            if class_label in class_distribution:
                class_distribution[class_label] += 1
            else:
                class_distribution[class_label] = 1

        return class_distribution


    def info_gain(self, current_gini, true_y, false_y):
        true_count = true_y.shape[0]
        false_count = false_y.shape[0]

        true_prop = float(true_count) / (float(true_count) + float(false_count))
        false_prop = 1 - true_prop

        gini_impurity = (true_prop * self.gini_impurity(true_y)) + (false_prop * self.gini_impurity(false_y))

        pure_split = False
        if gini_impurity == 0:
            pure_split = True
        return current_gini - gini_impurity, pure_split


    def classify(self, x, decision_node):
        # Once it get to the leaf node, just return the predictions
        if isinstance(decision_node, Leaf):
            return decision_node.predictions

        # If it gets here, it means it is a Decision node
        col = decision_node.split['col_index']
        val = decision_node.split['value']

        # Depending on the col value, recursively call this class until
        # you get a leaf node
        if x[col] >= val:
            return self.classify(x, decision_node.true_branch)
        else:
            return self.classify(x, decision_node.false_branch)


    def get_unique_val(self,X):
        unique_values = {}
        for col in self.sampled_features:
            # Find unique values in each column and split it
            unique_values[col] = set(X[:, col])
        return unique_values


    def get_random_features(self,X):
        total_features = len(X[0])
        # np.random.seed(0)
        if self.max_features == "sqrt":
            n = int(math.sqrt(total_features))
            self.sampled_features = np.random.choice(total_features,n)
        elif self.max_features == "auto":
            self.sampled_features = np.random.choice(total_features,total_features)
        elif self.max_features < total_features:
            self.sampled_features = np.random.choice(total_features,self.max_features)
        else:
            self.sampled_features = np.random.choice(total_features, total_features)

    def split_values(self,col,y):
        col = np.reshape(col, (col.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        x = np.concatenate((col, y), axis=1)

        threshold = 20
        # threshold = int(col.shape[0]*0.05)
        streak = {}
        values_to_split = []

        sorted_x = x[x[:, 0].argsort()]

        for i in sorted_x:
            # Streak is empty, it means either it was emptied because the streak
            # was broken or this is the first element of array
            if not streak:
                streak[i[1]] = 1
            else:
                last_label = list(streak.keys())[0]
                # If the current label matched the last label in dictionary
                # Just increment the count
                if i[1] == last_label:
                    streak[last_label] = streak[last_label] + 1
                # Else check if the occurences of that class are greater than threshold
                # and decide to split on that point or not
                else:
                    if streak[last_label] >= threshold:
                        values_to_split.append(i[0])

                    streak = {i[1]: 1}

        return values_to_split


    def predict(self, test_X):
        y_pred = []
        for i, x in enumerate(test_X):
            predictions = self.classify(x, self.root_node)
            predicted_class = max(predictions, key=lambda x: predictions[x])
            y_pred.append(predicted_class)

        return np.array(y_pred)



class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the split condition, and to the two child nodes.
    """

    def __init__(self, split, true_branch, false_branch):
        self.split = split
        self.true_branch = true_branch
        self.false_branch = false_branch


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., {180: 3}) -> class label "180" appeared
    3 times in the rows passed to this Leaf.
    """

    def __init__(self, y):
        predictions = {}
        for class_label in y:
            if class_label in predictions:
                predictions[class_label] += 1
            else:
                predictions[class_label] = 1

        self.predictions = predictions




class RandomForest:
    '''
    n_estimators = No. of Decision trees to build for this Random Forest
    sample_size = Sample size of the training data for each test (i.e for each Decision Tree)
    depth = Depth of the Decision tree
    max_features = ref:sklearn
    min_sample_leaf = ref:sklearn

    '''
    def __init__(self, n_estimators, sample_size, depth, max_features):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.depth = depth
        self.max_features = max_features


    def fit(self, X, y):
        '''
        Generate Decision trees for the training data
        :param X: np.array of training data
        :param y: np.array of training label corresponding to training data (X)
        :return:
        '''
        dTrees = []

        # Create decision trees equal to n_estimators
        for i in range(self.n_estimators):
            # Take a random sample from the training data
            sample_X, sample_y = self.random_sample(X,y)
            dTree = DecisionTree(self.depth,self.max_features)
            dTree.fit(sample_X, sample_y)
            dTrees.append(dTree)

        # Assign this collection of decision trees to a class variable
        self.dTrees = dTrees


    def predict(self, test):
        # Take each test point and run it through all the Decision trees created to get predictions
        y_predict = []
        for i, row in enumerate(test):
            predictions = []
            for j, dTree in enumerate(self.dTrees):
                predictions.append(dTree.predict([row])[0])

            label = max(set(predictions), key=predictions.count)
            y_predict.append(label)

        return np.array(y_predict)


    def random_sample(self,X,y):
        total = X.shape[0]
        percent = self.sample_size/100.0
        datapoints_to_take = int(percent*total)
        indices = np.random.choice(X.shape[0], datapoints_to_take)
        sample_X = X[indices, :]
        sample_y = y[indices]
        return sample_X,sample_y

    def score(self,y_pred,y_actual):
        count = 0.0
        for i in range(len(y_pred)):
            if y_pred[i] == y_actual[i]:
                count += 1

        return count/len(y_pred)

