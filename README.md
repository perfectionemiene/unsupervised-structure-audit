# Unsupervised Structure Audit

## Project Overview

This project investigates the underlying structure of two datasets using unsupervised machine learning techniques. The goal is to identify meaningful groups in the data without using reference labels during the initial clustering process.

The project uses the Wine and Digits datasets from Scikit-learn to explore dimensionality reduction, clustering algorithms, model evaluation, and machine learning pipeline implementation.

Reference labels were only revealed after the pre-reveal clustering predictions had been documented and committed to Git.

## Objectives

- Explore and preprocess datasets for unsupervised learning.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Compare different clustering algorithms.
- Evaluate clustering quality using internal and external evaluation metrics.
- Compare clustering performance with and without PCA.
- Build and persist a reusable Scikit-learn pipeline.
- Test that saved and reloaded pipeline predictions remain identical.
- Compare discovered clusters with reference labels after the commitment stage.

## Datasets

### 1. Wine Dataset

The Wine dataset contains 178 observations and 13 numerical features describing chemical properties of wine samples.

The dataset was examined for missing values, differences in feature scales, and skewness.

Preprocessing:

- No missing values were found.
- Features had different numerical ranges.
- Some features, including magnesium, malic acid, color intensity, and proline, showed noticeable positive skewness.
- StandardScaler was applied to give features a mean of zero and standard deviation of one.

### 2. Digits Dataset

The Digits dataset contains 1,797 handwritten digit images represented by 64 numerical features. Each image is an 8 × 8 pixel representation.

PCA was applied to reduce the original 64 features to 29 components while retaining approximately 95.48% of the variance.

## Methods and Algorithms

The following clustering algorithms were investigated:

- K-Means: Groups observations around cluster centroids.
- Ward Hierarchical Clustering: Builds clusters using a hierarchical merging approach.
- Gaussian Mixture Model (GMM): Models observations as a mixture of probability distributions.
- DBSCAN: Identifies dense regions and classifies observations outside those regions as noise.

The analysis also included PCA visualisations, explained variance analysis, K-Means inertia comparisons, hierarchical dendrograms, and DBSCAN parameter exploration.

## Wine Dataset Results

### PCA Findings

- First principal component: 36.20% explained variance.
- First two components: 55.41% cumulative explained variance.
- First five components: 80.16% cumulative explained variance.
- Ten components: approximately 96.17% cumulative explained variance.

### Clustering Evaluation

| Algorithm        |    ARI |    NMI | Homogeneity | Completeness | V-measure |
| ---------------- | -----: | -----: | ----------: | -----------: | --------: |
| K-Means          | 0.8975 | 0.8759 |      0.8788 |       0.8730 |    0.8759 |
| Ward             | 0.7899 | 0.7865 |      0.7904 |       0.7825 |    0.7865 |
| Gaussian Mixture | 0.8804 | 0.8609 |      0.8639 |       0.8580 |    0.8609 |
| DBSCAN           | 0.3291 | 0.4191 |      0.4104 |       0.4282 |    0.4191 |

K-Means achieved the strongest agreement with the reference labels, followed closely by Gaussian Mixture. Ward also identified meaningful structure, while DBSCAN performed less effectively.

DBSCAN's results were sensitive to its parameter settings. The selected configuration used `eps=2.2` and `min_samples=5`, producing two clusters and 55 noise observations.

The DBSCAN evaluation should be interpreted with this noise assignment in mind.

## Digits Dataset Results

K-Means was applied using ten clusters, both with and without PCA.

### PCA Findings

- Original feature count: 64.
- Reduced feature count: 29.
- Explained variance retained: 95.48%.

### Performance Comparison

| Measurement       |  Without PCA |     With PCA |
| ----------------- | -----------: | -----------: |
| Inertia           | 1,165,256.30 | 1,068,958.08 |
| Runtime (seconds) |       0.3812 |       0.1029 |
| ARI               |       0.6669 |       0.6625 |
| NMI               |       0.7394 |       0.7377 |
| Homogeneity       |       0.7353 |       0.7340 |
| Completeness      |       0.7436 |       0.7413 |
| V-measure         |       0.7394 |       0.7377 |

PCA reduced the runtime in this experiment while preserving nearly the same clustering evaluation scores.

The approach without PCA performed marginally better across the external evaluation metrics. However, the differences were small.

Inertia values from the two feature spaces are not directly comparable because PCA changes the representation and dimensionality of the data.

## Evaluation Metrics

The following metrics were used after revealing the reference labels:

- **Adjusted Rand Index (ARI):** Measures agreement between cluster assignments and reference classes, adjusted for chance.
- **Normalized Mutual Information (NMI):** Measures the shared information between clustering assignments and reference labels.
- **Homogeneity:** Measures whether each cluster contains observations predominantly from one reference class.
- **Completeness:** Measures whether observations belonging to the same reference class are grouped together.
- **V-measure:** The harmonic mean of homogeneity and completeness.

Silhouette scores were also used during the initial unsupervised analysis to assess cluster separation without reference labels.

Ordinary accuracy was not used because cluster identifiers do not necessarily correspond to reference class identifiers.

## Machine Learning Pipeline

A reusable Scikit-learn pipeline was implemented in `src/clustering.py`.

The pipeline consists of:

1. StandardScaler for feature standardisation.
2. PCA for dimensionality reduction.
3. K-Means for clustering.

The pipeline was trained on the Wine dataset and saved using Joblib at:

`models/wine_pipeline.joblib`

An automated test in `tests/test_pipeline.py` verifies that predictions remain identical after the trained pipeline is saved and reloaded.

## Project Structure

text
unsupervised-structure-audit/

── notebooks/
└── A1_unsupervised_audit.ipynb

── src/
├── **init**.py
└── clustering.py

── tests/
└── test_pipeline.py

── models/
└── wine_pipeline.joblib

── .gitignore
── README.md
── requirements.txt

## Installation and Usage

### 1. Clone the repository

```bash
git clone <repository-url>
cd unsupervised-structure-audit
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open the project in VS Code or Jupyter Notebook and run:

`notebooks/A1_unsupervised_audit.ipynb`

### 5. Run the automated test

```bash
python -m pytest -q
```

The test checks that the saved and reloaded pipeline produces identical predictions.

## Pre-Reveal Commitment Note

### Wine Dataset

Based on the exploratory analysis and clustering experiments, I predicted that the Wine dataset contained approximately three meaningful clusters.

My confidence in this prediction was moderate to low because the clustering algorithms produced similar but relatively modest silhouette scores. K-Means achieved 0.2849, Ward hierarchical clustering achieved 0.2774, and the three-component Gaussian Mixture model achieved 0.2844.

However, the Gaussian Mixture model selection criteria produced conflicting results: BIC favoured two components, while AIC favoured ten. This suggested that the underlying structure might not be straightforward.

I would reconsider my prediction if further evidence from the dendrogram, inertia curve, cluster stability, or model-selection criteria consistently supported a different number of groups.

### Digits Dataset

For the Digits dataset, I predicted approximately ten clusters, with moderate confidence.

This prediction was based on the expectation that handwritten digit images might form groups according to their visual patterns. However, handwriting variations and similarities between different digit shapes could make some groups difficult to separate.

K-Means was tested with ten clusters, both with and without PCA. PCA reduced the feature space from 64 to 29 components while retaining 95.48% of the variance.

I would reconsider this prediction if cluster evaluation, visual inspection, or comparison with the eventual reference labels indicated that the data did not separate into ten reasonably distinct groups.

### Reflection

These predictions were provisional and based entirely on unsupervised analysis. No reference class labels were used to guide the clustering decisions or evaluate the predictions at that stage.

After committing this note, the reference labels were revealed and used to evaluate how closely the discovered clusters corresponded to the actual classes.

---

## Conclusion

This project demonstrates how unsupervised learning techniques can uncover structure in datasets without initially relying on known class labels.

The Wine experiments showed that K-Means and Gaussian Mixture achieved strong agreement with the reference classes. The Digits experiments demonstrated that PCA reduced dimensionality and improved computational efficiency while maintaining similar clustering quality.

The project also highlights the importance of comparing internal and external evaluation metrics, investigating algorithm limitations, and preserving reproducible machine learning workflows through pipelines and automated tests.
