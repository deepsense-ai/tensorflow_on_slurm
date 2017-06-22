# CIFAR-10 Example

This directory contains an example job training a simple CNN model for CIFAR-10 on a Slurm cluster with distributed TensorFlow. Please note that the example is meant only to illustrate the usage of this package, the accuracy of the model could certainly be easily improved.

## Running the example

Firstly, create a virtualenv and install TensorFlow and tensorflow_on_slurm in it with:

```
virtualenv cifar-10-job-venv
source venv/bin/activate
pip install tensorflow
pip install git+https://github.com/deepsense-io/tensorflow_on_slurm
```

The virtualenv is necessary to make sure the packages are there on all the nodes taking part in the job. Now to run the example script on 5 nodes (4 workers + 1 parameter server) issue:

```
srun  -N 5 -n 5 -t 24:00:00 job.sh cifar-10-job-venv
```
