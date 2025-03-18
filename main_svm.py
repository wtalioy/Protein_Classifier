import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.svm import SVC
from Bio.PDB import PDBParser


class SVMModel:
# Todo
    """
        Initialize Support Vector Machine (SVM from sklearn) model.

        Parameters:
        - C (float): Regularization parameter. Default is 1.0.
        - kernel (str): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
    """

    """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self, C=1.0, kernel='rbf'):
        pass

    def train(self, train_data, train_targets):
        pass


    def evaluate(self, data, targets):
        pass

class SVMFromScratch:
    def __init__(self, lr=0.001, num_iter=200, c=0.01):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0
        self.C = c
        self.mean = None
        self.std = None

    def compute_loss(self, y, predictions):
        """
        SVM Loss function:
        hinge_loss = 1/2 * ||w||^2 + C * sum(max(0, 1 - y * z))
        """
        num_samples = y.shape[0]
        # 1. Hinge Loss: max(0, 1 - y * z)
        hinge_loss = np.maximum(0, 1 - y * predictions)
        hinge_loss_sum = np.sum(hinge_loss)  # sum
        # 2. Regularization term: 1/2 * ||w||^2
        regularization = 0.5 * np.sum(self.weights ** 2)
        total_loss = regularization + self.C * hinge_loss_sum
        loss = total_loss / num_samples
        return loss
    
    def standardize(self, X):
        return (X - self.mean) / self.std

    #  todo:
    def train(self, train_data, train_targets):
        X = np.array(train_data)
        y = np.array(train_targets)

        # Convert tags to 1 and -1
        y = np.where(y == 0, -1, 1)

        # Standardize Data
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-3   
        X = self.standardize(X)  

        num_samples, num_features = X.shape
        # Initialize weights and biases
        self.weights = np.zeros(num_features)
        self.bias = 0   
        # Gradient descent updates parameters
        for iteration in range(self.num_iter):
            for i in range(num_samples):  
                pass
                # ### Todo
                # ### Calculate the output of a svm model
                # svm_output = 
                # ### Calculate the gradient dw, db  (提示：根据y[i]与svm_output是否符号一致，分类讨论dw，db)
                
                # ### Update weights and bias
                # self.weights -= 
                # self.bias -= 

            
            if iteration % 10 == 0:
                predictions = np.dot(X, self.weights) + self.bias
                loss = self.compute_loss(y, predictions)
                print(f"Iteration {iteration}, Loss: {loss}")
        

    def predict(self, X):
        # sign 
        X = self.standardize(X)  
        svm_model = np.dot(X, self.weights) + self.bias
        predictions = np.sign(svm_model)  
        return predictions

    def evaluate(self, data, targets):
        X = np.array(data)
        y = np.array(targets)
        y = np.where(y == 0, -1, 1)
        predictions = self.predict(X)
        return np.mean(predictions == y)
    

def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    
    for task in range(1, 56):
        task_col = cast.iloc[:, task]

        #### Todo: Try to load data/target
        
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []
    
    ## Todo:Model Initialization 
    ## You can also consider other different settings
    # model = SVMModel(C=args.C,kernel=args.kernel)
    model = SVMFromScratch()


    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Training and Evaluation")
    parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    parser.add_argument('--kernel', type=str, default='rbf', help="Kernel type for SVM")
    args = parser.parse_args()
    main(args)

