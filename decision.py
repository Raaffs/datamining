import math

# Function to calculate Gini Impurity for a set of labels
def gini_impurity(labels):
    total = len(labels)
    if total == 0:
        return 0
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    gini = 1
    for count in label_counts.values():
        probability = count / total
        gini -= probability ** 2
    return gini

# Function to split a dataset based on a feature and a threshold
def split_data(dataset, feature_index, threshold):
    left_split = []
    right_split = []
    for row in dataset:
        if row[feature_index] <= threshold:
            left_split.append(row)
        else:
            right_split.append(row)
    return left_split, right_split

# Function to find the best split
def best_split(dataset):
    best_gini = float("inf")
    best_split_point = None
    best_left_split = None
    best_right_split = None
    
    num_features = len(dataset[0]) - 1  # Last column is the label
    
    for feature_index in range(num_features):
        thresholds = set(row[feature_index] for row in dataset)
        
        for threshold in thresholds:
            left_split, right_split = split_data(dataset, feature_index, threshold)
            
            # Skip if either split is empty
            if not left_split or not right_split:
                continue
            
            # Calculate the Gini impurity for the split
            left_labels = [row[-1] for row in left_split]
            right_labels = [row[-1] for row in right_split]
            gini_left = gini_impurity(left_labels)
            gini_right = gini_impurity(right_labels)
            
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
    labels = [row[-1] for row in dataset]
    
    # If the dataset is pure (all labels are the same), return a leaf node
    if len(set(labels)) == 1:
        return labels[0]
    
    # If max_depth is reached, return the most frequent label
    if max_depth is not None and depth >= max_depth:
        return max(set(labels), key=labels.count)
    
    # Find the best split
    best_split_point, left_split, right_split = best_split(dataset)
    
    # If no valid split found (cannot split further), return the most frequent label
    if best_split_point is None:
        return max(set(labels), key=labels.count)
    
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
dataset = [
    [2.7, 1.1, 'A'],
    [1.9, 1.8, 'A'],
    [3.0, 3.4, 'B'],
    [3.7, 3.8, 'B'],
    [2.2, 1.3, 'A'],
    [4.2, 3.9, 'B'],
]

# Build the decision tree (with max depth of 3)
tree = build_tree(dataset, max_depth=3)

# Print the tree structure
print("Decision Tree Structure:")
print(tree)

# Example predictions
test_data = [
    [2.5, 1.4],  # Expected to predict 'A'
    [3.5, 3.5],  # Expected to predict 'B'
]

for row in test_data:
    print(f"Prediction for {row}: {predict(tree, row)}")
