# Timbre Dissimilarity Metrics
A collection of metrics for evaluating timbre dissimilarity using the TorchMetrics API

## Installation
`pip install -e .`

## Usage
```
import timbremetrics

datasets = timbremetrics.list_datasets()
dataset = datasets[0] # get the first timbre dataset

# MAE between target dataset and pred embedding distances
metric = timbremetrics.TimbreMAE(
    margin=0.0, dataset=dataset, distance=timbremetrics.l1
)

# get numpy audio for the timbre dataset
audio = timbremetrics.get_audio(dataset)

# get arbitrary embeddings for the timbre dataset's audio
embeddings = net(audio)

# compute the metric
metric(embeddings)

```

## Metrics

The following metrics are implemented.

### Mean Squared Error

Gives the mean squared error between the upper triangles of the predicted distance matrix and target distance matrix:

![Mean squared error equation](https://latex.codecogs.com/png.latex?%5Ctext%7BMSE%7D%28D_X%2CD_Y%29%20%3D%20%5Cfrac%7B2%7D%7Bn%28n-1%29%7D%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%28D_X-D_Y%29%5E2)

### Mean Absolute Error

Gives the mean squared error between the upper triangles of the predicted distance matrix and target distance matrix:

![Mean absolute error equation](https://latex.codecogs.com/png.latex?%5Ctext%7BMAE%7D%28D_X%2CD_Y%29%20%3D%20%5Cfrac%7B2%7D%7Bn%28n-1%29%7D%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7CD_X-D_Y%7C)

### Item Rank Agreement

Gives the proportion of distances ranked per-item that match between the predicted distance matrix and target distance matrix.

![Item rank agreement equation](https://latex.codecogs.com/png.latex?%5Ctext%7BIRA%7D%28R_X%2CR_Y%29%20%3D%20%5Cfrac%7B1%7D%7Bn%5E2%20-%20n%7D%5Cleft%5B%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Cleft%28%201_%7B%5C%7B0%5C%7D%7D%28R_X_%7Bi%2Cj%7D-R_Y_%7Bi%2Cj%7D%29%20%5Cright%20%29%20-n%20%5Cright%5D)

Where ![idf](https://latex.codecogs.com/png.latex?1_A%28x%29) is the indicator function given by:

![Indicator function](https://latex.codecogs.com/png.latex?1_A%28x%29%20%3A%3D%20%5Cbegin%7Bcases%7D%201%20%5Cquad%20%5Ctext%7Bif%20%7D%20x%20%5Cin%20A%20%5C%5C%200%20%5Cquad%20%5Ctext%7Bif%20%7D%20x%20%5Cnotin%20A%20%5C%20%5Cend%7Bcases%7D)

and ![R_X](https://latex.codecogs.com/png.latex?R_X) & ![R_Y](https://latex.codecogs.com/png.latex?R_Y) are distances matrices ranked per item such that each row contains the ordinal distances from the corresponding item. We also provide a _top-k_ version which computes this metric considering only the closest _k_ items in each row.
