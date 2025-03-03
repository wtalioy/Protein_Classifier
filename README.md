## In-class Practice: LR Protein Classification

### Environment Setup:
1. **Python Environment:** Ensure you have Python installed (preferably Python 3.x).
2. **Dependencies:** Install required dependencies using pip:

```
pip install numpy pandas scikit-learn biopython
```

### Code Flow:
1. **Data Preprocessing:**
- The code preprocesses the data, loading protein structure diagrams and sequence information.
- If the `--ent` flag is provided, the data is loaded from a file using a feature engineering function `feature_extraction()` from fea.py. Otherwise, it loads from pre-existing files.
- The code reads a CAST file containing protein sequence information and a Numpy array containing diagrams.

2. **Model Initialization:**
- You should implement the **LRModel** class and **LRFromScratch** class.
- For LRModel class, perhaps you can explore different initialization settings,like Regularization parameter `C`.
- For the LRFromScratch class, you should implement LR model with gradient descent optimization **by your own code**. This means you should not use torch or any other deep learning library.

3. **Training and Evaluation:**
- The code trains the selected model on the training data and evaluates its performance on both training and test datasets.
- It partitions the dataset into training and testing sets for each task.
- The model's accuracy is printed for each dataset.

### Filling in the Blanks:
1. **LRModel Class:**
- Fill in the initialization, training, and evaluation methods for the Logistic Regression model.

2. **LRFromScratch Class:**
- Fill in the initialization, training, and evaluation methods for the Logistic Regression model of your own implementation.

3. **Data processing:**
- Read a CAST file containing protein sequence information and a Numpy array containing diagrams.Process them in training form.

### Running the Script:
- Execute the script `main.py` from the command line.
- You can provide arguments to customize  data loading method.


### Experimental Requirements:

1. **Complete Implementation of Protein Classification (Data Loading and LRModel) - 1.5 points**
    - Ensure the code effectively reads and preprocesses protein structure data and sequences.
    - Implement LR model from sklearn library.


2. **Complete Implementation of LRFromScratch - 1.5 points**
    - Implement LRFromScratch model with methods of gradient descent.
    - Pay attention to the loss function format you choose, since the labels here are in {0,1}.
    
**Total Score: 3 points**

### Other exploration:
1. **Analysis and Discussion on the Impact of Regularization Coefficients and other initialization settings**
    - Analyze how varying the regularization coefficient (`C`) affects the model's performance and generalization.
    - Except Regularization Coefficients,choose 1 or 2 other settings you want to explore.
2. **Feature Engineering**
    - Provide insights on extracting useful features from protein structure data or utilizing feature selection methods to reduce dimensionality.

  

### Submission:
Submit both the code and the running results in a single zip archive named **"学号_姓名_课堂练习1.zip"**.

**ddl: 2025/3/24 23:59**
