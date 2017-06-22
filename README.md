# Tensorflow on Slurm

This package makes it easier to run distributed TensorFlow jobs on slurm clusters. It contains functions for parsing the Slurm environment variables in order to create configuration for distributed TF.

## Prerequisites

You need to have TensorFlow installed. All the examples were tested with TensorFlow 1.0.1, but other versions also have a good chance of working correctly.

### Installation

To install execute the following on the command line:
```
git clone
cd tensorflow_on_slurm
sudo pip install .
```

## Usage

A complete usage example using the CIFAR-10 dataset is included in the examples directory.

However if you just want to dive in you can paste the following snippet into your script:

```python
import tensorflow as tf
from tensorflow_on_slurm import tf_config_from_slurm

cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
cluster_spec = tf.train.ClusterSpec(cluster)
server = tf.train.Server(server_or_cluster_def=cluster_spec,
                         job_name=my_job_name,
                         task_index=my_task_index)
```
## Issues

It's possible that our tests don't cover all the corner cases about the names of the Slurm nodes etc. If you happen to spot some bugs please don't hesitate to file an issue here on github.

## Contributing
Pull request are more than welcome. If you'd like to add some new functionality don't forget to write unit tests for it and make sure you don't break any currently working tests (see below on how to run the tests)

## Tests
To run the tests issue:

```
python tensorflow_on_slurm/tests/test.py
``` 

