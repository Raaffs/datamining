from itertools import combinations

# Function to calculate the support of itemsets
def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions)

# Function to generate frequent itemsets using the Apriori algorithm
def apriori(transactions, min_support):
    # Step 1: Generate 1-itemsets
    itemsets = set(item for transaction in transactions for item in transaction)
    frequent_itemsets = {frozenset([item]): calculate_support([item], transactions) for item in itemsets}
    
    # Step 2: Prune non-frequent 1-itemsets
    frequent_itemsets = {itemset: support for itemset, support in frequent_itemsets.items() if support >= min_support}
    
    # Step 3: Iteratively generate k-itemsets and prune non-frequent ones
    all_frequent_itemsets = dict(frequent_itemsets)  # Store all frequent itemsets
    k = 2  # Start with 2-itemsets
    while True:
        # Generate candidate k-itemsets from (k-1)-itemsets
        candidate_itemsets = set()
        frequent_itemset_list = list(frequent_itemsets.keys())
        for i in range(len(frequent_itemset_list)):
            for j in range(i + 1, len(frequent_itemset_list)):
                candidate = frequent_itemset_list[i] | frequent_itemset_list[j]  # Union of two itemsets
                if len(candidate) == k:  # Only consider itemsets of size k
                    candidate_itemsets.add(candidate)
        
        # Step 4: Calculate support for candidate itemsets
        candidate_support = {itemset: calculate_support(itemset, transactions) for itemset in candidate_itemsets}
        
        # Step 5: Prune non-frequent itemsets
        frequent_itemsets = {itemset: support for itemset, support in candidate_support.items() if support >= min_support}
        
        if not frequent_itemsets:  # No frequent itemsets found, stop
            break
        
        # Add the frequent itemsets to the list of all frequent itemsets
        all_frequent_itemsets.update(frequent_itemsets)
        
        k += 1  # Move to the next size itemsets

    return all_frequent_itemsets

# Example transaction dataset
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'butter'],
    ['bread', 'jam']
]

# Run the Apriori algorithm with minimum support threshold
min_support = 0.4
frequent_itemsets = apriori(transactions, min_support)

# Print the frequent itemsets and their support values
for itemset, support in frequent_itemsets.items():
    print(f"Itemset: {set(itemset)}, Support: {support:.2f}")
