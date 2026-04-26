import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans


FILE_PATH = "./Flow_CAB_10_student_810904023.csv"

# 1) Skip the first title row, and use the next row as the header
df = pd.read_csv(FILE_PATH, skiprows=1, header=0, encoding="latin-1")

# 2) Drop the first column (it's the scenario id: 1..60)
df = df.iloc[:, 1:]

# 3) Make sure everything is numeric
df = df.apply(pd.to_numeric, errors="coerce")

# 4) Optional safety: drop any completely empty columns
df = df.dropna(axis=1, how="all")

print("Final shape:", df.shape)   # should print (60, 20)
print(df.head(60))

scaler = StandardScaler()
X = scaler.fit_transform(df.values)

print("Scaled X shape:", X.shape)


MIN_SAMPLES = 4  # typical small value

nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES)
nbrs.fit(X)

distances, _ = nbrs.kneighbors(X)
kth_dist = np.sort(distances[:, -1])  # distance to k-th nearest neighbor

plt.figure(figsize=(7, 5))
plt.plot(np.arange(len(kth_dist)), kth_dist, marker=".")
plt.title(f"k-distance curve (k = min_samples = {MIN_SAMPLES})")
plt.xlabel("Points sorted by k-distance")
plt.ylabel("Distance to k-th nearest neighbor")
plt.show()

EPS = 2.7   # based on the plot; replace if the data set got changed


dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
db_labels = dbscan.fit_predict(X)

n_noise = int((db_labels == -1).sum())
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

print("Chosen eps:", EPS)
print("DBSCAN noise points:", n_noise)
print("DBSCAN clusters (excluding noise):", n_clusters)
print("DBSCAN labels:", db_labels)

keep_mask = db_labels != -1

df_no_noise = df.loc[keep_mask].copy()
X_no_noise = X[keep_mask]

print("Rows before:", len(df))
print("Rows after noise removal:", len(df_no_noise))

noise_row_positions = np.where(~keep_mask)[0].tolist()
print("Noise row positions removed (indexes are from 0):", noise_row_positions)


K_MAX = min(10, len(X_no_noise))
ks = list(range(1, K_MAX + 1))
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_no_noise)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(ks, inertias, marker="o")
plt.title("Elbow plot (KMeans after DBSCAN noise removal)")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia (within-cluster SSE)")
plt.xticks(ks)
plt.show()

print(pd.DataFrame({"k": ks, "inertia": inertias}))

BEST_K = 3  # set based on the elbow plot

kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=50)
km_labels = kmeans.fit_predict(X_no_noise)

print("Chosen k:", BEST_K)
print("Cluster sizes:\n", pd.Series(km_labels).value_counts().sort_index())

results = pd.DataFrame({
    "dbscan_label": db_labels,
    "is_noise": (db_labels == -1)
})

# cluster label exists only for non-noise rows
results["kmeans_cluster"] = np.nan
results.loc[keep_mask, "kmeans_cluster"] = km_labels

results.to_csv("task3_labels_all_rows.csv", index=False, encoding="utf-8-sig")

df_no_noise_out = df_no_noise.copy()
df_no_noise_out["cluster"] = km_labels
df_no_noise_out.to_csv("task3_clustered_no_noise.csv", index=False, encoding="utf-8-sig")

print("Saved:")
print("- task3_labels_all_rows.csv")
print("- task3_clustered_no_noise.csv")
