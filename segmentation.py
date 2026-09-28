"""
Task 4: Netflix Content Segmentation
Unsupervised learning (K-Means) to group Netflix titles into meaningful clusters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Dataset.csv"
RANDOM_STATE = 42
TOP_COUNTRIES = 10
TOP_GENRES = 15
K_RANGE = range(4, 11)

sns.set_theme(style="whitegrid")


# Prepare numerical and categorical features.
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset="show_id").reset_index(drop=True)

# Numerical features
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year.fillna(df["release_year"])
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["movie_minutes"] = np.where(df["type"] == "Movie", df["duration_num"], 0)
df["tv_seasons"] = np.where(df["type"] == "TV Show", df["duration_num"], 0)

numeric = df[["release_year", "year_added", "movie_minutes", "tv_seasons"]]
numeric = pd.DataFrame(
    StandardScaler().fit_transform(numeric), columns=numeric.columns
)

# Categorical features
df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
top_countries = df["primary_country"].value_counts().head(TOP_COUNTRIES).index
df["primary_country"] = df["primary_country"].where(
    df["primary_country"].isin(top_countries), "Other"
)

type_rating_country = pd.get_dummies(
    df[["type", "rating", "primary_country"]], dtype=float
)

genres = df["listed_in"].str.split(",").explode().str.strip()
top_genres = genres.value_counts().head(TOP_GENRES).index
genre_matrix = (
    pd.crosstab(genres.index, genres)[top_genres].clip(upper=1).astype(float)
)
genre_matrix.columns = [f"genre_{g}" for g in genre_matrix.columns]
genre_matrix = genre_matrix.reindex(df.index, fill_value=0)

X = pd.concat([numeric, type_rating_country, genre_matrix], axis=1)
print(f"Feature matrix shape: {X.shape}")


# Apply K-Means clustering and compare candidate cluster counts.
inertia_scores = []
silhouette_scores = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    inertia_scores.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

best_k = list(K_RANGE)[int(np.argmax(silhouette_scores))]
print(f"Optimal number of clusters (highest silhouette score): {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(K_RANGE), inertia_scores, marker="o")
ax[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia")
ax[1].plot(list(K_RANGE), silhouette_scores, marker="o", color="darkorange")
ax[1].set(title="Silhouette Score", xlabel="Number of clusters (k)", ylabel="Score")
plt.tight_layout()
plt.savefig("optimal_k.png", dpi=150)
plt.show()

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
df["cluster"] = kmeans.fit_predict(X)


# Summarize the resulting content groups.
print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nSample titles per cluster:")
for cluster_id in sorted(df["cluster"].unique()):
    titles = df.loc[df["cluster"] == cluster_id, "title"].head(5).tolist()
    print(f"\nCluster {cluster_id}:", titles)


# Visualize the clusters with a two-dimensional PCA projection.
pca = PCA(n_components=2, random_state=RANDOM_STATE)
components = pca.fit_transform(X)
df["pca1"], df["pca2"] = components[:, 0], components[:, 1]

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, x="pca1", y="pca2", hue="cluster", palette="tab10", s=18, alpha=0.7
)
plt.title("Netflix Content Clusters (PCA Projection)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("clusters_pca.png", dpi=150)
plt.show()


# Profile the characteristics of each cluster.
df["genre_list"] = df["listed_in"].str.split(",")

profile = df.groupby("cluster").agg(
    titles=("show_id", "count"),
    movie_share=("type", lambda s: (s == "Movie").mean()),
    avg_release_year=("release_year", "mean"),
    avg_movie_minutes=("movie_minutes", lambda s: s[s > 0].mean()),
    avg_tv_seasons=("tv_seasons", lambda s: s[s > 0].mean()),
    top_rating=("rating", lambda s: s.mode().iloc[0]),
    top_country=("primary_country", lambda s: s.mode().iloc[0]),
)
profile["top_genres"] = df.groupby("cluster")["genre_list"].apply(
    lambda s: ", ".join(
        s.explode().str.strip().value_counts().head(3).index
    )
)
print("\nCluster profile:")
print(profile.round(2).to_string())
profile.round(2).to_csv("cluster_profile.csv")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="cluster", hue="type", palette="Set2")
plt.title("Content Type by Cluster")
plt.tight_layout()
plt.savefig("cluster_type_distribution.png", dpi=150)
plt.show()

df.drop(columns=["genre_list", "pca1", "pca2"]).to_csv(
    "netflix_clustered.csv", index=False
)
