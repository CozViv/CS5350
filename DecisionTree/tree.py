import math
import pandas as pd

def entropy(labels):
    total = len(labels)
    value_counts = {}
    for label in labels:
        if label not in value_counts:
            value_counts[label] = 0
        value_counts[label] += 1

    entropy_value = 0
    for count in value_counts.values():
        probability = count / total
        entropy_value -= probability * math.log2(probability)
    return entropy_value

def majority_error(labels):
    total = len(labels)
    if total == 0:
        return 0
    value_counts = {}
    for label in labels:
        if label not in value_counts:
            value_counts[label] = 0
        value_counts[label] += 1

    majority_class_count = max(value_counts.values())
    return 1 - (majority_class_count / total)

def gini_index(labels):
    total = len(labels)
    value_counts = {}
    for label in labels:
        if label not in value_counts:
            value_counts[label] = 0
        value_counts[label] += 1

    gini_value = 1
    for count in value_counts.values():
        probability = count / total
        gini_value -= probability ** 2
    return gini_value

def calculate_gain(data, feature_index, target, f):
    if f == 'majority_error':
        total_error = majority_error(data[target])
    elif f == 'gini':
        total_error = gini_index(data[target])
    elif f == 'entropy':
        total_error = entropy(data[target])
    
    values = data.iloc[:, feature_index].unique()
    weighted_error = 0
    for value in values:
        subset = data[data.iloc[:, feature_index] == value]
        if f == 'majority_error':
            subset_error = majority_error(subset[target])
        elif f == 'gini':
            subset_error = gini_index(subset[target])
        elif f == 'entropy':
            subset_error = entropy(subset[target])
        
        weighted_error += (len(subset) / len(data)) * subset_error
    
    gain = total_error - weighted_error
    #print(gain)
    return gain

def build_tree(data, target, features, f='entropy', depth=0, max_depth=None):
    labels = data[target]
    
    # 1
    if len(set(labels)) == 1:
        return labels.iloc[0]
    if len(features) == 0 or (max_depth is not None and depth >= max_depth):
        return labels.mode()[0]
    
    #2
    #2.2
    gains = [calculate_gain(data, i, target, f) for i in range(len(features))]
    best_feature_index = gains.index(max(gains))
    best_feature = features[best_feature_index]
    #print("best feature:", best_feature)
    #2.1
    tree = {best_feature: {}}

    #print("data.iloc: ",data.iloc[:, best_feature_index])
    feature_values = data.iloc[:, best_feature_index].unique()
    remaining_features = [f for f in features if f != best_feature]
    
    #print("feature values:", feature_values)
    #print("remaining features:", remaining_features)
    #3
    for value in feature_values:
        #3.2
        subset = data[data.iloc[:, best_feature_index] == value]
        subset = subset.drop(columns=[best_feature])
        #3.3
        subtree = build_tree(subset, target, remaining_features, f, depth + 1, max_depth)
        #3.1
        tree[best_feature][value] = subtree

    #4
    return tree

def decision_tree_from_csv(file_path, target, columns, f='entropy', max_depth=None):
    data = pd.read_csv(file_path, header=None)
    data = data.astype(str)
    data.columns = columns
    features = [col for col in data.columns if col != target]
    tree = build_tree(data, target, features, f, max_depth=max_depth)
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree

    #feature = next(iter(tree))
    feature = list(tree.keys())[0]
    feature_value = instance[feature]

    if feature_value in tree[feature]:
        return predict(tree[feature][feature_value], instance)
    else:
        # Handle unseen feature values
        #print(feature, feature_value)
        return "Unknown"

def calculate_error_rate(predictions, actual_labels):
    incorrect_predictions = sum(pred != actual for pred, actual in zip(predictions, actual_labels))
    error_rate = incorrect_predictions / len(actual_labels)
    return error_rate

def test_decision_tree(tree, test_data, target):
    predictions = []
    actual_labels = test_data[target].tolist()
    for _, row in test_data.iterrows():
        prediction = predict(tree, row)
        predictions.append(prediction)
    
    error_rate = calculate_error_rate(predictions, actual_labels)
    #print(predictions)
    #print(actual_labels)
    return predictions, error_rate

# Training data
file_path = 'car/train.csv'
target_column = "label"
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]

# Testing on training data
test_file_path = 'car/train.csv'
test_data = pd.read_csv(test_file_path, header=None)
test_data = test_data.astype(str)
test_data.columns = columns

average_e_train = 0
for i in range(1,7):
    md = i
    tree_entropy = decision_tree_from_csv(file_path, target_column, columns, f='entropy', max_depth=md)
    #print(tree_entropy)
    predictions_entropy, error_rate_entropy = test_decision_tree(tree_entropy, test_data, target_column)
    print(f"Accuracy for training set with Entropy at depth of {md}:", 1 - error_rate_entropy)
    average_e_train += (1-error_rate_entropy)
average_e_train = average_e_train/6
print("\n")

average_g_train = 0
for i in range(1,7):
    md = i
    tree_gini = decision_tree_from_csv(file_path, target_column, columns, f='gini', max_depth=md)
    predictions_gini, error_rate_gini = test_decision_tree(tree_gini, test_data, target_column)
    print(f"Accuracy for training set with Gini at depth of {md}", 1 - error_rate_gini)
    average_g_train += (1-error_rate_gini)
average_g_train = average_g_train/6
print("\n")

average_m_train = 0
for i in range(1,7):
    md = i
    tree_me = decision_tree_from_csv(file_path, target_column, columns, f='majority_error', max_depth=md)
    predictions_me, error_rate_me = test_decision_tree(tree_me, test_data, target_column)
    print(f"Accuracy for training set with ME at depth of {md}", 1 - error_rate_me)
    average_m_train += 1-error_rate_me
average_m_train = average_m_train/6

# Testing on testing data
test_file_path = 'car/test.csv'
test_data = pd.read_csv(test_file_path, header=None)
test_data = test_data.astype(str)
test_data.columns = columns

print("\n")
average_e_test = 0
for i in range(1,7):
    md = i
    tree_entropy = decision_tree_from_csv(file_path, target_column, columns, f='entropy', max_depth=md)
    #print(tree_entropy)
    predictions_entropy, error_rate_entropy = test_decision_tree(tree_entropy, test_data, target_column)
    print(f"Accuracy for testing set with Entropy at depth of {md}:", 1 - error_rate_entropy)
    average_e_test += 1-error_rate_entropy
average_e_test = average_e_test/6
print("\n")

average_g_test = 0
for i in range(1,7):
    md = i
    tree_gini = decision_tree_from_csv(file_path, target_column, columns, f='gini', max_depth=md)
    predictions_gini, error_rate_gini = test_decision_tree(tree_gini, test_data, target_column)
    print(f"Accuracy for testing set with Gini at depth of {md}", 1 - error_rate_gini)
    average_g_test += 1-error_rate_gini
average_g_test = average_g_test/6
print("\n")

average_m_test = 0
for i in range(1,7):
    md = i
    tree_me = decision_tree_from_csv(file_path, target_column, columns, f='majority_error', max_depth=md)
    predictions_me, error_rate_me = test_decision_tree(tree_me, test_data, target_column)
    print(f"Accuracy for testing set with ME at depth of {md}", 1 - error_rate_me)
    average_m_test += 1 - error_rate_me
average_m_test = average_m_test / 6

print("Average accuracy on training set for entropy: ", average_e_train)
print("Average accuracy on training set for GI: ", average_g_train)
print("Average accuracy on training set for ME: ", average_m_train)

print("Average accuracy on test set for entropy: ", average_e_test)
print("Average accuracy on test set for GI: ", average_g_test)
print("Average accuracy on test set for ME: ", average_m_test)