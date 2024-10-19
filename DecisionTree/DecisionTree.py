import math
import pandas as pd

class DecisionTree:
    def __init__(self, f='entropy', max_depth=None):
        self.f = f
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, labels):
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

    def majority_error(self, labels):
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

    def gini_index(self, labels):
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

    def calculate_gain(self, data, feature_index, target):
        if self.f == 'majority_error':
            total_error = self.majority_error(data[target])
        elif self.f == 'gini':
            total_error = self.gini_index(data[target])
        elif self.f == 'entropy':
            total_error = self.entropy(data[target])
        
        values = data.iloc[:, feature_index].unique()
        weighted_error = 0
        for value in values:
            subset = data[data.iloc[:, feature_index] == value]
            if self.f == 'majority_error':
                subset_error = self.majority_error(subset[target])
            elif self.f == 'gini':
                subset_error = self.gini_index(subset[target])
            elif self.f == 'entropy':
                subset_error = self.entropy(subset[target])
            
            weighted_error += (len(subset) / len(data)) * subset_error
        
        gain = total_error - weighted_error
        return gain

    def build_tree(self, data, target, features, depth=0):
        labels = data[target]

        if len(set(labels)) == 1:
            return labels.iloc[0]
        
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            mode_label = labels.mode()
            return mode_label[0] if len(mode_label) > 0 else None

        gains = [self.calculate_gain(data, i, target) for i in range(len(features))]
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
            tree[best_feature]['<= ' + str(median)] = self.build_tree(subset_less_equal, target, remaining_features, depth + 1)
            tree[best_feature]['> ' + str(median)] = self.build_tree(subset_greater, target, remaining_features, depth + 1)
        else:
            # If the feature is categorical, split by unique values
            feature_values = data[best_feature].unique()
            remaining_features = [f for f in features if f != best_feature]

            for value in feature_values:
                subset = data[data[best_feature] == value].copy()
                subset = subset.drop(columns=[best_feature])
                tree[best_feature][value] = self.build_tree(subset, target, remaining_features, depth + 1)

        return tree

    def fit(self, data, target, features):
        self.tree = self.build_tree(data, target, features)
        return self.tree

    def predict(self, tree, instance):
        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        feature_value = instance[feature]

        if isinstance(tree[feature], dict):
            for condition, subtree in tree[feature].items():
                if "<=" in condition:
                    median_value = float(condition.split("<=")[1].strip()) # Isolate median
                    if float(feature_value) <= median_value:
                        return self.predict(subtree, instance)
                elif ">" in condition:
                    median_value = float(condition.split(">")[1].strip()) # Isolate median
                    if float(feature_value) > median_value:
                        return self.predict(subtree, instance)
                elif feature_value == condition: # If not a <= or >, split on values
                    return self.predict(subtree, instance)

        return "Unknown" # Return unknown if there is no prediction (did not appear in training tree)

    def calculate_error_rate(self, predictions, actual_labels):
        incorrect_predictions = sum(pred != actual for pred, actual in zip(predictions, actual_labels))
        error_rate = incorrect_predictions / len(actual_labels)
        return error_rate

    def test_decision_tree(self, test_data, target):
        predictions = [self.predict(self.tree, instance) for _, instance in test_data.iterrows()]
        actual_labels = test_data[target].tolist()
        error_rate = self.calculate_error_rate(predictions, actual_labels)
        return error_rate
