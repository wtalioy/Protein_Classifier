# Report on LR Model Training and Evaluation

## Model Hyperparameters

- **LRModel Configuration:**
  - Max Iterations: 100
  - Regularization Constant (C): 2.5

- **LRFromScratch Configuration:**
  - Max Iterations: 200
  - Learning Rate: 5e-4
  - Regularization Constant (C): 2.5

## Training and Test Results

- **LRModel:**
    ```bash
    Training accuracy: 0.9024405918831598
    Testing accuracy: 0.9734976682645816
    ```

- **LRFromScratch:**
    ```bash
    Training accuracy: 0.8677558810611198
    Testing accuracy: 0.9433237662044374
    ```

## Analysis

- LRModel achieved a higher testing accuracy (0.9735) compared to LRFromScratch (0.9433). The measure it takes optimizing the model parameters using a more sophisticated optimization algorithm (LBFGS) and regularization techniques, which likely contributed to its superior performance, while LRFromScratch used a simpler gradient descent approach (thereby learning rate hyperparameter is added).

- Regularization constant (C) has minimal impact on the performance of both models.

- 100 iterations for LRModel are sufficient for convergence due to the efficient optimization algorithm used, while LRFromScratch required 200 iterations or more to achieve a similar level of convergence (especially for the first three classification tasks). Increasing learning rate for LRFromScratch may seem plausible, but it leads to overshooting the optimal solution and causes divergence.