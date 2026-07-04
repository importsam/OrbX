# OrbX

OrbX is a Python toolkit for orbital capacity analysis. It clusters orbits based on geometrical similarity using Keplerian elements (excluding mean motion), generates representative synthetic orbits for each cluster, and computes a density score that helps quantify how tightly packed different orbital neighbourhoods are.

This project was developed as part of the following paper:

> **OrbX: A Framework for Orbital Capacity Management**  
> Sam White, Samya Bagchi, Yasir Latif  
> 12th Annual Space Traffic Management Conference, Austin TX, February 2026  
> [Paper link](https://github.com/importsam/OrbX/blob/main/OrbX__A_Framework_for_Orbital_Capacity_Characterization.pdf)

OrbX is designed for workflows such as:

- grouping similar resident space objects into orbital neighbourhoods/clusters
- identifying representative reference orbits for a cluster
- finding the most geometrically isolated orbit within a neighbourhood
- comparing cluster density

---

## Installation

```bash
pip install orbx
```

**Requirements:**

- Python `3.9+`

> **Note:** [orekit](https://gitlab.orekit.org/orekit-labs/python-wrapper) is required for `density()` and `synthetic_orbit()`. Orekit is not pip installable, so install it via Conda if needed:
>
> ```bash
> conda install -c conda-forge orekit
> ```

---

## Quick Start

```python
import pandas as pd
from orbx import cluster, synthetic_orbit, density

df = pd.DataFrame(
    {
        "line1": [...],
        "line2": [...],
    }
)

# 1. Cluster orbits
labels = cluster(df)
df["label"] = labels

# 2. Remove noise (recommended, optional)
clustered_df = df[df["label"] != -1].copy()

# 3. Generate synthetic orbits
synthetic_df = synthetic_orbit(
    clustered_df,
    mode=["frechet", "max_separation"]
)

# 4. Score cluster density
density_df = density(clustered_df)
```

---

## What OrbX Does

OrbX exposes three core functions:

### `cluster()`

Clusters satellite orbits based on geometric similarity, using [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) and Keplerian elements extracted from [two-line elements](https://en.wikipedia.org/wiki/Two-line_element_set) (TLEs).

### `synthetic_orbit()`

Generates one or more synthetic orbits for each non-noise cluster. Two modes are supported:

- `"frechet"` — computes a Fréchet mean orbit, which acts like a centroid in a non-Euclidean orbital metric space
- `"max_separation"` — finds an orbit inside the cluster region that maximises separation from existing members, highlighting available space within the neighbourhood

### `density()`

Computes a density score for each labelled cluster by measuring dispersion around the cluster's Fréchet mean orbit. Useful for ranking or comparing how concentrated orbital neighbourhoods are within.

---

## Input Format

A minimum OrbX workflow may start with a DataFrame containing TLE rows:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "line1": [...],
        "line2": [...],
    }
)
```

After clustering, add the returned labels back onto the DataFrame in a `label` column before calling `synthetic_orbit()` or `density()`.

---

## API Reference

### `cluster(df, min_samples=3, min_cluster_size=2, verbose=False)`

Groups similar TLEs into orbital neighbourhoods and returns one cluster label per input row.

**Arguments:**


| Argument           | Type        | Description                                                                                                                    |
| ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `df`               | `DataFrame` | Must contain `line1` and `line2` TLE columns. Each row is one object.                                                          |
| `min_samples`      | `int`       | Controls clustering conservativeness. Higher values require stronger local support, which can increase noise points (`-1`).    |
| `min_cluster_size` | `int`       | Minimum cluster size HDBSCAN will return. Higher values suppress small clusters and favour larger, more stable neighbourhoods. |
| `verbose`          | `bool`      | If `True`, shows internal clustering output and warnings.                                                                      |


**Tuning notes:**

- Lower `min_samples` and `min_cluster_size` → more, smaller clusters
- Higher `min_samples` and `min_cluster_size` → fewer clusters, more noise points
- Both parameters default to the values used in the paper

**Returns:** NumPy array of integer cluster labels aligned to input row order. `-1` = noise / unclustered.

---

### `synthetic_orbit(df, mode="max_separation", n_samples=5000, verbose=False)`

Generates synthetic orbits for each cluster in a labelled DataFrame. Two modes are supported:

- `"frechet"` — computes a Fréchet mean orbit, which represents a centroid in a non-Euclidean orbital metric space
- `"max_separation"` — finds an orbit inside the cluster that maximises separation from existing cluster members

**Arguments:**


| Argument    | Type            | Description                                                                                                   |
| ----------- | --------------- | ------------------------------------------------------------------------------------------------------------- |
| `df`        | `DataFrame`     | Must contain `line1`, `line2`, and `label` columns.                                                           |
| `mode`      | `str` or `list` | `"frechet"`, `"max_separation"`, or `["frechet", "max_separation"]` to run both.                              |
| `n_samples` | `int`           | Initial candidate samples for `"max_separation"` search. Higher values improve quality, but increase runtime. |
| `verbose`   | `bool`          | If `True`, shows optimiser warnings and progress.                                                             |


**Returns:** DataFrame of synthetic TLE rows with columns `line1`, `line2`, `label`, and `synthetic_type`.

---

### `density(df, label_column="label", verbose=False)`

Computes a density score for each cluster by measuring the spread of member orbits around the cluster's Fréchet mean orbit.

**Arguments:**


| Argument       | Type        | Description                                                              |
| -------------- | ----------- | ------------------------------------------------------------------------ |
| `df`           | `DataFrame` | Must contain `line1`, `line2`, and a cluster label column.               |
| `label_column` | `str`       | Name of the column containing cluster IDs. Defaults to `"label"`.        |
| `verbose`      | `bool`      | If `True`, shows optimiser diagnostics while computing the Fréchet mean. |


**Returns:** DataFrame with one row per cluster and columns `label` and `density`. Values should be interpreted relative to other clusters in the same analysis — lower density means a more dispersed neighbourhood, higher means more tightly packed.

---

## Outputs


| Function            | Returns                                                    |
| ------------------- | ---------------------------------------------------------- |
| `cluster()`         | NumPy array of integer cluster labels                      |
| `synthetic_orbit()` | DataFrame with `line1`, `line2`, `label`, `synthetic_type` |
| `density()`         | DataFrame with `label` and `density` per cluster           |


---

## Demo


| Demo                                                                 | Description                                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [Unique Orbits (k-NN) Cesium Model](https://orbx.spaceprotocol.org/) | 3D Cesium visualisation of orbital neighbourhoods and uniqueness across regimes |


---

## References


| Ref | Resource                   | Link                                                                                                                                                             |
| --- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1] | scikit-learn `HDBSCAN` API | [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) |
| [2] | OrbX conference paper      | [https://github.com/importsam/OrbX/blob/main/OrbX__A_Framework_for_Orbital_Capacity_Characterization.pdf](https://github.com/importsam/OrbX/blob/main/OrbX__A_Framework_for_Orbital_Capacity_Characterization.pdf)         |
| [3] | Two-line element set (TLE) | [https://en.wikipedia.org/wiki/Two-line_element_set](https://en.wikipedia.org/wiki/Two-line_element_set)                                                         |


