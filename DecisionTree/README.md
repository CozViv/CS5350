# CS5350
This is a decision tree creator and tester. To use this, import DecisionTree from DecisionTree into your
file, then create a list for the columns, a target column, and a feature list, which is columns without target column.

Create a tree using DecistionTree(f='function you're wanting to use', max_depth=int)
train a tree using the tree you created with the training data as the first, the target column, and the features in order.
You can test the tree by calling test_decision_tree on your created tree with the data you want to test it on and the
target column in the paramaters.

To use the parts for the assignment, call
python carDecisionTree.py
python bankDecisionTree.py

which will print out the expected results for the assignment