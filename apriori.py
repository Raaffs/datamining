import itertools

# Function to calculate support for itemsets
def calculate_support(dataset, itemset):
    itemset = set(itemset)
    count = 0
    for transaction in dataset:
        if itemset.issubset(set(transaction)):
            count += 1
    return count / len(dataset)

# Function to generate candidate itemsets of length 1
def generate_candidates(dataset):
    candidates = set()
    for transaction in dataset:
        for item in transaction:
            candidates.add(frozenset([item]))
    return candidates

# Function to generate candidate itemsets of length k
def generate_candidate_k_itemsets(frequent_itemsets, k):
    candidates = set()
    for itemset1, itemset2 in itertools.combinations(frequent_itemsets, 2):
        union_itemset = itemset1.union(itemset2)
        if len(union_itemset) == k:
            candidates.add(union_itemset)
    return candidates

# Function to perform the Apriori algorithm
def apriori(dataset, min_support):
    # Step 1: Convert the dataset into a list of transactions
    transactions = [set(transaction) for transaction in dataset]
    
    # Step 2: Generate candidates of length 1
    candidates = generate_candidates(transactions)
    
    # Step 3: Generate frequent itemsets
    frequent_itemsets = []
    
    k = 1
    while candidates:
        # Calculate support for each candidate
        itemset_support = {itemset: calculate_support(transactions, itemset) for itemset in candidates}
        
        # Prune non-frequent itemsets (those that don't meet the min_support)
        frequent_itemsets_k = {itemset for itemset, support in itemset_support.items() if support >= min_support}
        
        if not frequent_itemsets_k:
            break
        
        frequent_itemsets.append(frequent_itemsets_k)

        # Generate candidates for the next iteration (next size)
        k += 1
        candidates = generate_candidate_k_itemsets(frequent_itemsets_k, k)
    
    return frequent_itemsets

# Example dataset: each transaction is a list of items
dataset = [
    ['apple', 'banana', 'milk', 'bread'],
    ['apple', 'banana', 'milk'],
    ['apple', 'bread', 'milk'],
    ['banana', 'milk', 'bread'],
    ['apple', 'banana', 'milk', 'orange', 'bread'],

 
]

# Minimum support threshold (e.g., 0.5 means itemsets must appear in at least 50% of transactions)
min_support = 0.5

# Run Apriori algorithm
frequent_itemsets = apriori(dataset, min_support)

# Print frequent itemsets found by Apriori
for k, itemsets in enumerate(frequent_itemsets, start=1):
    print(f"Frequent Itemsets of size {k}:")
    for itemset in itemsets:
        print(f"  {set(itemset)}")
