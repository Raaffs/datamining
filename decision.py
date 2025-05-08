import numpy as np
import pandas as pd

# Function to calculate Gini Impurity for a set of labels
def gini_impurity(labels):
    total = len(labels)
    if total == 0:
        return 0
    # Count the frequency of each label
    label_counts = labels.value_counts()
    gini = 1
    for count in label_counts:
        probability = count / total
        gini -= probability ** 2
    return gini

# Function to split the data based on a feature and threshold
def split_data(dataset, feature_index, threshold):
    left_split = dataset[dataset.iloc[:, feature_index] <= threshold]
    right_split = dataset[dataset.iloc[:, feature_index] > threshold]
    return left_split, right_split

# Function to find the best split
def best_split(dataset):
    best_gini = float("inf")
    best_split_point = None
    best_left_split = None
    best_right_split = None
    
    num_features = dataset.shape[1] - 1  # Last column is the label
    
    for feature_index in range(num_features):
        thresholds = dataset.iloc[:, feature_index].unique()  # Get unique values for the feature
        
        for threshold in thresholds:
            left_split, right_split = split_data(dataset, feature_index, threshold)
            
            # Skip if either split is empty
            if left_split.empty or right_split.empty:
                continue
            
            # Calculate the Gini Impurity for the split
            gini_left = gini_impurity(left_split.iloc[:, -1])
            gini_right = gini_impurity(right_split.iloc[:, -1])
            
            # Weighted Gini Impurity
            weighted_gini = (len(left_split) / len(dataset)) * gini_left + (len(right_split) / len(dataset)) * gini_right
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split_point = (feature_index, threshold)
                best_left_split = left_split
                best_right_split = right_split
                
    return best_split_point, best_left_split, best_right_split

# Function to build the decision tree recursively
def build_tree(dataset, max_depth=None, depth=0):
    labels = dataset.iloc[:, -1]
    
    # If the dataset is pure (all labels are the same), return a leaf node
    if labels.nunique() == 1:
        return labels.iloc[0]
    
    # If max_depth is reached, return the most frequent label
    if max_depth is not None and depth >= max_depth:
        return labels.mode().iloc[0]
    
    # Find the best split
    best_split_point, left_split, right_split = best_split(dataset)
    
    # If no valid split found (cannot split further), return the most frequent label
    if best_split_point is None:
        return labels.mode().iloc[0]
    
    # Recursively build left and right subtrees
    left_tree = build_tree(left_split, max_depth, depth + 1)
    right_tree = build_tree(right_split, max_depth, depth + 1)
    
    # Return the tree in a structure: (feature, threshold, left subtree, right subtree)
    return (best_split_point, left_tree, right_tree)

# Function to make predictions using the decision tree
def predict(tree, row):
    if not isinstance(tree, tuple):
        return tree  # Leaf node (label)
    
    feature_index, threshold = tree[0]
    if row[feature_index] <= threshold:
        return predict(tree[1], row)  # Left subtree
    else:
        return predict(tree[2], row)  # Right subtree

# Example dataset: Last column is the label
data = {
    'Feature1': [2.7, 1.9, 3.0, 3.7, 2.2, 4.2],
    'Feature2': [1.1, 1.8, 3.4, 3.8, 1.3, 3.9],
    'Label': ['A', 'A', 'B', 'B', 'A', 'B']
}

# Convert to pandas DataFrame
dataset = pd.DataFrame(data)

# Build the decision tree (with max depth of 3)
tree = build_tree(dataset, max_depth=3)

# Print the tree structure
print("Decision Tree Structure:")
print(tree)

# Example predictions (test data)
test_data = [
    [2.5, 1.4],  # Expected to predict 'A'
    [3.5, 3.5],  # Expected to predict 'B'
]

# Make predictions for the test data
for row in test_data:
    print(f"Prediction for {row}: {predict(tree, row)}")
