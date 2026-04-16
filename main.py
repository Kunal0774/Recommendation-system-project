# =========================================
# COMPLETE CONTENT-BASED RECOMMENDATION SYSTEM
# =========================================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Load Data
# -------------------------------
df = pd.read_csv("data.csv")

# Strip whitespace from all column names (fixes 'item_n ' → 'item_n')
df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully!\n")
print("Columns in dataset:", df.columns.tolist())

# -------------------------------
# Step 2: Create User-Item Matrix
# -------------------------------
user_item_matrix = df.pivot_table(
    index='user_id',
    columns='item_n',
    values='rating'
).fillna(0)

print("\nUser-Item Matrix:\n")
print(user_item_matrix)

# -------------------------------
# Step 3: Compute Similarity
# -------------------------------
item_similarity = cosine_similarity(user_item_matrix.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print("\nItem Similarity Matrix:\n")
print(item_similarity_df)

# -------------------------------
# Step 4: Recommendation Function
# -------------------------------
def recommend_items(item_name, top_n=3):
    if item_name not in item_similarity_df.columns:
        print(f"\n❌ Item '{item_name}' not found!")
        print("Available items:", list(item_similarity_df.columns))
        return None

    similar_items = item_similarity_df[item_name].sort_values(ascending=False)
    similar_items = similar_items.drop(item_name)

    return similar_items.head(top_n)

# -------------------------------
# Step 5: User Interaction
# -------------------------------
print("\nAvailable Movies:")
for item in user_item_matrix.columns:
    print("-", item)

user_input = input("\nEnter a movie name: ").strip()  # .strip() handles accidental spaces

recommendations = recommend_items(user_input)

# -------------------------------
# Step 6: Show Results
# -------------------------------
if recommendations is not None:
    print(f"\n🎯 Top Recommendations for '{user_input}':\n")
    for item, score in recommendations.items():
        print(f"  {item}  (Similarity: {round(score, 2)})")