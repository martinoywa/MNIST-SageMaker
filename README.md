# MNIST-SageMaker
Build, Train, Deploy, and Query an MNIST Image Classifier on AWS


# Steps
1. Hyperparameter Optimization.
- Objective: Minimize Test Loss.
- Perfom hyperparameter search that satisfies this objective.

2. Training.
- Train a custom CNN based model using the best hyperparameters from HPO step above.

3. Deployment.

- Method 1 (Basic Model Loader then Query in Notebook)

- Method 2 (Extensive Process then Query with a Lambda Function). Tagged as _v2 in the code.


Build with ❤️ by [martinoywa](https://github.com/martinoywa)
