# Thesis Topic: Clustering Patient Data Using Network Embeddings

The mine of healthcare data has a treasury of enriched information. This data can provide advancements in the healthcare system. Despite this, the data is heterogeneous and complex in nature, making analysis a critical task. To overcome this problem, machine learning is a promising field of research. Today’s machine learning provides enormous advancements in many fields, and it aligns with technology to integrate a better healthcare system. This thesis is a small step toward supporting machine learning in conducting healthcare research. It explores network embedding techniques by representing healthcare data as graphs, with nodes representing patients, encounters, treatments, and edges reflecting their relationships.

This master’s thesis aims to evaluate the effectiveness of a Network Embedding (NE) such as Node2Vec in clustering patient data. A comparative analysis with traditional tabular clustering methods is conducted to assess performance in terms of cluster coherence and clinical relevance.

Using synthetic healthcare data generated with Synthea, patients are clustered using:
	•	Tabular methods: K-Means and DBSCAN
	•	Graph-based methods: Node2Vec embeddings + K-Means and DBSCAN

Clusters are evaluated using Jaccard similarity based on encounter class and treatment descriptions.

The key contributions of this thesis are as follows:
1. Graph Construction: Constructing a graph from a synthetic relational database, capturing the effective representation of relationships between data points.
2. Clustering through Graph Embedding: Applying graph embedding networks to perform clustering, where node representations are learned based on information propagation within the graph.
3. Cluster Evaluation: Clusters will form, but evaluating them is essential to assess their quality and clinical relevance. Findings are based on assessing their ability to group similar patients into cohesive clusters. Thus, patients are grouped based on encounter classes and evaluated using appropriate evaluation metrics for this purpose.

⸻
# Results

DBSCAN: Network Embedding vs Tabular Approach

| Dataset Size (Patients) | NE - Encounter Class | NE - Treatments | Tabular - Encounter Class | Tabular - Treatments |
|--------------------------|----------------------|-----------------|---------------------------|----------------------|
| 100 | 0.62 | 0.34 | 0.68 | 0.57 |
| 250 | 0.68 | 0.53 | 0.73 | 0.47 |
| 500 | 0.67 | 0.49 | 0.79 | 0.64 |
| 1000 | 0.64 | 0.71 | 0.79 | 0.60 |


K-Means: Network Embedding vs Tabular Approach

| Dataset Size (Patients) | NE - Encounter Class | NE - Treatments | Tabular - Encounter Class | Tabular - Treatments |
|--------------------------|----------------------|-----------------|---------------------------|----------------------|
| 100 | 0.58 | 0.28 | 0.89 | 0.80 |
| 250 | 0.60 | 0.29 | 0.75 | 0.49 |
| 500 | 0.62 | 0.32 | 0.74 | 0.46 |
| 1000 | 0.63 | 0.34 | 0.77 | 0.47 |

⸻
### Evaluation Insight
 
Clusters were evaluated using the Jaccard similarity index to measure similarity between patients within the same cluster. The results indicate that high similarity scores were not achieved. This suggests a potential mismatch between the clustering structure learned through network embeddings and the evaluation metric used. The findings highlight the complexity and multi-dimensional nature of healthcare data, where structural similarity captured by graph embeddings may not fully align with feature-based similarity metrics.

⸻
## Key Modules

```bash
git clone https://github.com/RamanBaghel/Clustering-Patient-Data-Using-Network-Embedding.git
cd Clustering-Patient-Data-Using-Network-Embedding
pip install -r requirements.txt
```

⸻
## Run the Project

```bash
bash run.sh
```
⸻
## Modules

- `db.py` – Core experimental pipeline and clustering execution
- `run.sh` – End-to-end execution script
- `CSV.py` – Data preparation utilities
- `logplot.py` – Visualization of clustering results





