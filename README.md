# Pre-Reveal Commitment Note

## Unsupervised Structure Audit

### 1. Wine Dataset

Based on the exploratory analysis and clustering experiments, I predict that the Wine dataset contains approximately three meaningful clusters.

My confidence in this prediction is moderate to low because the clustering algorithms produced similar but relatively modest silhouette scores. K-Means achieved 0.2849, Ward hierarchical clustering achieved 0.2774, and the three-component Gaussian Mixture model achieved 0.2844.

However, the Gaussian Mixture model selection criteria produced conflicting results: BIC favoured two components, while AIC favoured ten. This suggests that the underlying structure may not be straightforward.

I would reconsider my prediction if further evidence from the dendrogram, inertia curve, cluster stability or model-selection criteria consistently supported a different number of groups.

### 2. Digits Dataset

For the Digits dataset, I predict approximately ten clusters, with moderate confidence.

This prediction is based on the expectation that handwritten digit images may form groups according to their visual patterns. However, handwriting variations and similarities between different digit shapes may make some groups difficult to separate.

K-Means was tested with ten clusters, both with and without PCA. PCA reduced the feature space from 64 to 29 components while retaining 95.48% of the variance.

I would reconsider this prediction if cluster evaluation, visual inspection or comparison with the eventual reference labels indicated that the data does not separate into ten reasonably distinct groups.

### 3. Reflection

These predictions are provisional and are based entirely on unsupervised analysis. No reference class labels have been used to guide the clustering decisions or evaluate the predictions at this stage.

After committing this note, I will reveal the reference labels and evaluate how closely the discovered clusters correspond to the actual classes.
