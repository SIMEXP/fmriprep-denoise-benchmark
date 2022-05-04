# Source of the files


Gordon 333: https://sites.wustl.edu/petersenschlaggarlab/parcels-19cwpgu/

```python
# parse gordon atlas
from fmriprep_denoise.metrics import compute_pairwise_distance

centroids = pd.read_csv("file/path/", sep='\t')
centroids = centroids['Centroid (MNI)'].str.split(' ', n=-1, expand=True)
centroids = centroids.applymap(lambda x : float(x))

pairwise_distance = compute_pairwise_distance(centroids)

pairwise_distance.to_csv(Path.cwd().parents[0] / \
    "inputs/atlas-gordon333_nroi-333_desc-distance.tsv", sep='\t', index=False)
```