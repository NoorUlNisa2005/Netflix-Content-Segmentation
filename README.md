# Netflix Content Segmentation

Grouping Netflix titles into meaningful clusters using unsupervised machine learning (K-Means).

## Overview

This project segments Netflix movies and TV shows into distinct content groups based on their numerical and categorical attributes. The optimal number of clusters is selected automatically using the Elbow method and the Silhouette score, and the resulting clusters are visualized and interpreted.

## Dataset

`Dataset.csv` contains 8,790 titles and 10 columns:

| Column | Description |
|---|---|
| `show_id` | Unique title ID |
| `type` | Movie or TV Show |
| `title` | Title name |
| `director` | Director(s) |
| `country` | Country of production |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Content rating (e.g. TV-MA, PG-13) |
| `duration` | Minutes (movies) or seasons (TV shows) |
| `listed_in` | Genres |

## Workflow

1. **Prepare features**
   - Numerical: `release_year`, `year_added`, `movie_minutes`, `tv_seasons` (standardized with `StandardScaler`).
   - Categorical: `type`, `rating`, primary country (top 10, rest grouped as "Other") one-hot encoded, and the 15 most frequent genres multi-hot encoded.
2. **Apply clustering**: K-Means for k = 4 to 10. The best k is the one with the highest Silhouette score.
3. **Identify content groups**: cluster sizes and sample titles per cluster.
4. **Visualize clusters**: 2D PCA projection, Elbow/Silhouette plots and type distribution per cluster.
5. **Interpret clusters**: per-cluster profile (movie share, average release year, average duration, top rating, top country, top genres).

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Place `Dataset.csv` in the same folder as the script (or change `DATA_PATH` in the script).
2. Run:

```bash
python netflix_content_segmentation.py
```

## Configuration

Adjustable constants at the top of the script:

| Constant | Default | Purpose |
|---|---|---|
| `DATA_PATH` | `Dataset.csv` | Path to the dataset |
| `RANDOM_STATE` | `42` | Reproducibility |
| `TOP_COUNTRIES` | `10` | Number of countries kept as separate features |
| `TOP_GENRES` | `15` | Number of genres kept as features |
| `K_RANGE` | `range(4, 11)` | Cluster counts evaluated |

`K_RANGE` starts at 4 on purpose. With k = 2 the model only separates Movies from TV Shows, which is not a meaningful segmentation.

## Output Files

| File | Description |
|---|---|
| `optimal_k.png` | Elbow and Silhouette plots |
| `clusters_pca.png` | PCA scatter plot of the clusters |
| `cluster_type_distribution.png` | Movie vs TV Show count per cluster |
| `cluster_profile.csv` | Summary statistics for each cluster |
| `netflix_clustered.csv` | Original data with the assigned `cluster` label |

## Results

The Silhouette score selected **5 clusters**:

| Cluster | Titles | Description |
|---|---|---|
| 0 | 3,274 | Modern movies, mainly International Movies, Dramas and Comedies |
| 1 | 580 | Older classic movies (average release year about 1988), mostly R-rated |
| 2 | 2,289 | Movies with a shorter average runtime, released around 2014 to 2015 |
| 3 | 252 | Long-running TV series (about 5.7 seasons on average): TV Comedies, TV Dramas, Kids' TV |
| 4 | 2,395 | Recent TV shows, mostly 1 to 2 seasons: International TV Shows, TV Dramas |

Cluster assignments can vary slightly if the dataset, feature settings or library versions change.

## Project Structure

```
.
├── Dataset.csv
├── netflix_content_segmentation.py
└── README.md
```
