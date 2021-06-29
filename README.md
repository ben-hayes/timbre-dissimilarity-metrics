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
