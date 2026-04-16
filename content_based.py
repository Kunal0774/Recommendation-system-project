# ===============================
# CONTENT-BASED RECOMMENDATION
# ===============================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load Data
df = pd.read_csv("data.csv")

print("Dataset:\n", df)

# Step 2: Create User-Item Matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

print("\nUser-Item Matrix:\n", user_item_matrix)

# Step 3: Compute Item Similarity
item_similarity = cosine_similarity(user_item_matrix.T)

# Convert to DataFrame
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=user_item_matrix.columns,
                                  columns=user_item_matrix.columns)

print("\nItem Similarity Matrix:\n", item_similarity_df)

# Step 4: Recommendation Function
def recommend_items(item_id, top_n=2):
    similar_items = item_similarity_df[item_id].sort_values(ascending=False)
    
    # Remove itself
    similar_items = similar_items.drop(item_id)
    
    return similar_items.head(top_n)

# Step 5: Test
print("\nRecommended items for item 101:")
print(recommend_items(101))