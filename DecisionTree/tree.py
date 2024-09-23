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

    if len(set(labels)) == 1:
        return labels.iloc[0]

    if len(features) == 0 or (max_depth is not None and depth >= max_depth):
        if len(labels) == 0:
            return None
        mode_label = labels.mode()
        if len(mode_label) > 0:
            return mode_label[0]
        else:
            return None

    gains = [calculate_gain(data, i, target, f) for i in range(len(features))]
    best_feature_index = gains.index(max(gains))
    best_feature = features[best_feature_index]
    tree = {best_feature: {}}

    feature_type = data[best_feature].dtype

    if pd.api.types.is_numeric_dtype(feature_type):
        # Create subsets based on median
        median = data[best_feature].median()
        subset_less_equal = data[data[best_feature] <= median]
        subset_greater = data[data[best_feature] > median]
        subset_less_equal = subset_less_equal.drop(columns=[best_feature])
        subset_greater = subset_greater.drop(columns=[best_feature])

        remaining_features = [f for f in features if f != best_feature]

        # Recursively build the tree for each subset
        tree[best_feature]['<= ' + str(median)] = build_tree(subset_less_equal, target, remaining_features, f, depth + 1, max_depth)
        tree[best_feature]['> ' + str(median)] = build_tree(subset_greater, target, remaining_features, f, depth + 1, max_depth)
    else:
        # If the feature is categorical, split by unique values
        feature_values = data[best_feature].unique()
        remaining_features = [f for f in features if f != best_feature]

        for value in feature_values:
            subset = data[data[best_feature] == value].copy()
            subset = subset.drop(columns=[best_feature])

            subtree = build_tree(subset, target, remaining_features, f, depth + 1, max_depth)
            tree[best_feature][value] = subtree

    return tree




def decision_tree_from_csv(file_path, target, columns, f='entropy', max_depth=None, types=None):
    data = pd.read_csv(file_path, header=None, names=columns)

    # Apply the correct data types if provided
    if types:
        data = data.astype(types)

    features = [col for col in data.columns if col != target]
    tree = build_tree(data, target, features, f, max_depth=max_depth)
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]

    feature_value = instance[feature]

    if isinstance(tree[feature], dict):
        for condition, subtree in tree[feature].items():
            if "<=" in condition:
                median_value = float(condition.split("<=")[1].strip()) # Isolate median
                if float(feature_value) <= median_value:
                    return predict(subtree, instance)
            elif ">" in condition:
                median_value = float(condition.split(">")[1].strip()) # Isolate median
                if float(feature_value) > median_value:
                    return predict(subtree, instance)
            elif feature_value == condition: # If not a <= or >, split on values
                return predict(subtree, instance)

    return "Unknown" # Return unknown if there is no prediction (did not appear in training tree)


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
    #print(f"Accuracy for training set with Entropy at depth of {md}:", 1 - error_rate_entropy)
    average_e_train += (1-error_rate_entropy)
average_e_train = average_e_train/6
print("\n")

average_g_train = 0
for i in range(1,7):
    md = i
    tree_gini = decision_tree_from_csv(file_path, target_column, columns, f='gini', max_depth=md)
    predictions_gini, error_rate_gini = test_decision_tree(tree_gini, test_data, target_column)
    #print(f"Accuracy for training set with Gini at depth of {md}", 1 - error_rate_gini)
    average_g_train += (1-error_rate_gini)
average_g_train = average_g_train/6
print("\n")

average_m_train = 0
for i in range(1,7):
    md = i
    tree_me = decision_tree_from_csv(file_path, target_column, columns, f='majority_error', max_depth=md)
    predictions_me, error_rate_me = test_decision_tree(tree_me, test_data, target_column)
    #print(f"Accuracy for training set with ME at depth of {md}", 1 - error_rate_me)
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

print("\n")

# Training data
file_path_bank_train = 'bank/train.csv'
target_column_bank = "label"
columns_bank = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"]
types = {
    "age":  int,
    "job": 'category',
    "marital": 'category',
    "education": 'category',
    "default": 'category',
    "balance": int,
    "housing": 'category',
    "loan": 'category',
    "contact": 'category',
    "day": int,
    "month": 'category',
    "duration": int,
    "campaign": int,
    "pdays": int,
    "previous": int,
    "poutcome": 'category',
    "label": 'category'
}
train_data_bank = test_data_bank = pd.read_csv(file_path_bank_train, header=None)
train_data_bank.columns = columns_bank

file_path_bank_test = 'bank/test.csv'
test_data_bank  = test_data_bank = pd.read_csv(file_path_bank_test, header=None)
test_data_bank.columns = columns_bank

print("Part A------------") # Treating unknown as value
average_e_train = 0
average_g_train = 0
average_m_train = 0
average_e_test = 0
average_g_test = 0
average_m_test = 0
for i in range(1,17):
    print(i)
    md = i
    tree_bank_e = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='entropy', max_depth=md, types=types)
    tree_bank_g = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='gini', max_depth=md, types=types)
    tree_bank_m = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='majority_error', max_depth=md, types=types)


    _, error_rate_bank_train_e = test_decision_tree(tree_bank_e, train_data_bank, target_column_bank)
    _, error_rate_bank_train_g = test_decision_tree(tree_bank_g, train_data_bank, target_column_bank)
    _, error_rate_bank_train_m = test_decision_tree(tree_bank_m, train_data_bank, target_column_bank)
    _, error_rate_bank_test_e = test_decision_tree(tree_bank_e, test_data_bank, target_column_bank)
    _, error_rate_bank_test_g = test_decision_tree(tree_bank_g, test_data_bank, target_column_bank)
    _, error_rate_bank_test_m = test_decision_tree(tree_bank_m, test_data_bank, target_column_bank)

    average_e_train += 1 - error_rate_bank_train_e
    average_g_train += 1 - error_rate_bank_train_g
    average_m_train += 1 - error_rate_bank_train_m
    average_e_test += 1 - error_rate_bank_test_e
    average_g_test += 1 - error_rate_bank_test_g
    average_m_test += 1 - error_rate_bank_test_m

average_e_train = average_e_train/16
average_g_train = average_g_train/16
average_m_train = average_m_train/16
average_e_test = average_e_test/16
average_g_test = average_g_test/16
average_m_test = average_m_test/16

print("Average for training set on Entropy tree: ", average_e_train)
print("Average for training set on GI tree: ", average_g_train)
print("Average for training set on ME tree: ", average_m_train)
print("Average for testing set on Entropy tree: ", average_e_test)
print("Average for testing set on GI tree: ", average_g_test)
print("Average for testing set on ME tree: ", average_m_test)

print("\n")

# Proccess data
for column in columns_bank:
    most_common_value = train_data_bank[column].mode()[0]
    train_data_bank[column].replace("unknown", most_common_value, inplace=True)
for column in columns_bank:
    most_common_value = test_data_bank[column].mode()[0]
    test_data_bank[column].replace("unknown", most_common_value, inplace=True)


print("Part B------------") # Treating unknown as most common
average_e_train = 0
average_g_train = 0
average_m_train = 0
average_e_test = 0
average_g_test = 0
average_m_test = 0
for i in range(1,17):
    print(i)
    md = i
    tree_bank_e = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='entropy', max_depth=md, types=types)
    tree_bank_g = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='gini', max_depth=md, types=types)
    tree_bank_m = decision_tree_from_csv(file_path_bank_train, target_column_bank, columns_bank, f='majority_error', max_depth=md, types=types)


    _, error_rate_bank_train_e = test_decision_tree(tree_bank_e, train_data_bank, target_column_bank)
    _, error_rate_bank_train_g = test_decision_tree(tree_bank_g, train_data_bank, target_column_bank)
    _, error_rate_bank_train_m = test_decision_tree(tree_bank_m, train_data_bank, target_column_bank)
    _, error_rate_bank_test_e = test_decision_tree(tree_bank_e, test_data_bank, target_column_bank)
    _, error_rate_bank_test_g = test_decision_tree(tree_bank_g, test_data_bank, target_column_bank)
    _, error_rate_bank_test_m = test_decision_tree(tree_bank_m, test_data_bank, target_column_bank)

    average_e_train += 1 - error_rate_bank_train_e
    average_g_train += 1 - error_rate_bank_train_g
    average_m_train += 1 - error_rate_bank_train_m
    average_e_test += 1 - error_rate_bank_test_e
    average_g_test += 1 - error_rate_bank_test_g
    average_m_test += 1 - error_rate_bank_test_m

average_e_train = average_e_train/16
average_g_train = average_g_train/16
average_m_train = average_m_train/16
average_e_test = average_e_test/16
average_g_test = average_g_test/16
average_m_test = average_m_test/16

print("Average for training set on Entropy tree: ", average_e_train)
print("Average for training set on GI tree: ", average_g_train)
print("Average for training set on ME tree: ", average_m_train)
print("Average for testing set on Entropy tree: ", average_e_test)
print("Average for testing set on GI tree: ", average_g_test)
print("Average for testing set on ME tree: ", average_m_test)