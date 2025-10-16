# Causal Post-Processing

This repository provides the official source code for the paper **"Causal Post-Processing of Predictive Models"**.

In this work, we introduce causal post-processing (CPP) techniques to refine predictive scores using experimental data to improve intervention decisions. The primary algorithm proposed is the **Causal Residual Forest**, which corrects a model's predictions by learning the causal treatment effect not captured by the predictive scores.

The core algorithms are implemented in the `causal_tree` directory.

## Quick Start

The core algorithms are implemented in Cython for performance. To compile them, navigate to the `causal_tree` directory and run the following command:

```sh
python setup.py build_ext --inplace
```

We provide a Jupyter notebook in `examples/simulation_example.ipynb` that demonstrates the usage of the CPP algorithms.

This is the best place to start to understand the methodology and code.

## Replicating Paper Results

You can replicate the experimental results from the paper using the `run.py` script.

### Simulation Study

To run a single simulation with the settings used in the paper (e.g., $\rho=0.7,\sigma_{\epsilon}^2=1.6$), run:

```sh
python simulation/run.py --rho 0.7 --var_noise 1.6 --seed 1
```

To use cross-validation (CV) for model selection, run:

```sh
python simulation/cross_validation.py --pretune_file pretuned.csv --rho 0.7 --var_noise 1.6 --seed 2
```

**Note**: 

* This command runs only one simulation instance. To replicate the full set of results, we recommend running multiple simulations in parallel on a High-Performance Computing (HPC) cluster.
* The `pretune_file` contains CV results from a separate dataset (`seed=101`). For large datasets, we use these pre-tuned parameters of each method to reduce computation time.

### Empirical Study

To replicate the results on the Criteo dataset, run:

```sh
python criteo/run.py --data_file criteo-uplift-v2.1.csv --target f9 --seed 1
```

**Note**: You will need to download the Criteo dataset and provide the local path to the data file.
