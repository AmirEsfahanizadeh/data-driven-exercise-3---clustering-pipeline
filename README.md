Exercise 3 Report: Scenario Clustering with DBSCAN and K-
Means
1. Objective
The goal of Exercise 3 is to cluster a finite set of demand scenarios into a small number
of groups. First, DBSCAN is used to identify and remove noisy (outlier) scenarios. Then,
K-Means is applied to the remaining scenarios. The number of K-Means clusters is
selected using the Elbow method (inertia plot).
2. Dataset Preparation
The input file is a scenario table where each row represents one scenario and each
column represents a demand dimension (customer index). The raw CSV file contained an
extra descriptive row and a scenario-id column. To obtain the required 60×20 matrix
(only numerical demand values), the following steps were applied:
• Read the CSV using skiprows=1 so the descriptive first row is skipped.
• Drop the first column (scenario id 1..60) and keep only the 20 demand columns.
• Convert values to numeric and remove any empty columns/rows (safety for
clustering).
After these steps, the matrix shape was verified as 60 rows (scenarios) and 20 columns
(demand dimensions).
3. Feature Scaling
Because DBSCAN and K-Means rely on Euclidean distances, all features were
standardized using StandardScaler. This transforms each column to have mean 0 and
standard deviation 1. Scaling prevents large-magnitude columns from dominating
distance calculations and makes density and centroid computations comparable across
dimensions.
4. DBSCAN Noise Detection
DBSCAN requires two parameters: min_samples (the minimum number of neighbors to
define a dense region) and eps (the neighborhood radius). min_samples was set to 4. To
select eps, the k-distance curve was plotted, where each point is the distance from a
scenario to its 4th nearest neighbor (after scaling), sorted in ascending order. The ‘knee’
or sharp increase in this curve indicates a transition from dense regions to sparse/noisy
points.
Figure 1 shows the k-distance curve used for selecting eps.
Figure 1. k-distance curve for DBSCAN (k = min_samples = 4).
Based on Figure 1, eps was chosen as 2.7 (in standardized distance units), which lies
around the main knee before the largest jump in distances.
5. DBSCAN Results and Noise Removal
After running DBSCAN with eps = 2.7 and min_samples = 4, each scenario received a
label. Label −1 denotes noise scenarios; non-negative integers denote DBSCAN clusters.
The obtained results were:
• Scaled input matrix shape: 60 × 20
• Noise scenarios (label −1): 15
• DBSCAN clusters excluding noise: 3
• Remaining scenarios for K-Means: 45
The 15 noise scenarios were removed from further clustering, leaving 45 scenarios for K-
Means.
6. K-Means Clustering and Elbow Method
K-Means requires specifying the number of clusters k. To select k, the Elbow method
was applied by running K-Means for k = 1..10 on the noise-free data and recording
inertia (within-cluster sum of squared distances). Inertia always decreases as k increases;
the elbow is chosen where the improvement begins to diminish.
Figure 2 shows the Elbow plot computed after DBSCAN noise removal.
Figure 2. Elbow plot (inertia vs k) for K-Means after DBSCAN noise removal.
The inertia values showed a large reduction from k=1→2 and k=2→3, followed by
relatively smaller reductions afterward. Therefore, k=3 was selected.
7. Final Clustering Output
K-Means was executed with k = 3 on the remaining 45 scenarios. The resulting cluster
sizes were:
• Cluster 0: 15 scenarios
• Cluster 1: 15 scenarios
• Cluster 2: 15 scenarios
The final output of this exercise is a cluster label for each non-noise scenario, plus
identification of noise scenarios. These cluster memberships are used in the next exercise
to build cluster-wise uncertainty sets (e.g., min/max ‘box’ bounds per cluster).
8. Reproducibility Notes
Fixed random_state values were used in K-Means to make the results repeatable. The
DBSCAN eps choice was justified using the k-distance curve, and the K-Means k choice
was justified using the inertia elbow plot. All steps were implemented in Python using
scikit-learn.


