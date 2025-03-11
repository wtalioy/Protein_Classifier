import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser


class LRModel(LogisticRegression):
    # todo:
    """
        Initialize Logistic Regression (from sklearn) model.

    """

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self):
        super.__init__(
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver='lbfgs',
            max_iter=100,
            multi_class='auto',
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None,
        )

    def train(self, train_data, train_targets):
        self.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        self.score(data, targets)


class LRFromScratch:
    # todo:
    def __init__(
        self,
        tol=1e-4,
        C=1.0,
        max_iter=100,
        lr=1e-4
    ):
        self.tol = tol
        self.C = C
        self.max_iter = max_iter
        self.lr = lr


    def init_weight(self):
        self.weight = np.random.randn(self.feature_size)
    
    def h(self, x):
        x = np.array([1, x])
        return 1 / (1 + np.exp(-self.weight.dot(x)))
    
    def loss(self, train_data, train_targets):
        return -sum((y * np.log(self.h(x)) + (1 - y) * np.log(1 - self.h(x))) for x, y in zip(train_data, train_targets))
    
    def gradient(self, train_data, train_targets):
        return sum((y - self.h(x)) * x for x, y in zip(train_data, train_targets))
            

    def train(self, train_data, train_targets):
        assert train_data.shape[0] == train_targets.shape[0]
        self.feature_size = train_data.shape[-1]
        self.init_weight()
        for t in range(self.max_iter):
            curr_loss = self.loss(train_data, train_targets)
            gr = self.gradient(train_data, train_targets)
            if np.max(np.abs(gr) <= self.tol):
                return self
            self.weight -= self.lr * gr
        return self
        
    
    def evaluate(self, data, targets):
        assert self.weight is not None and data.shape[0] == targets.shape[0]
        corr = 0
        for x, y in zip(data, targets):
            output = 1 if self.h(x) > 0.5 else 0
            corr += (output == y)
        return corr / data.shape[0]


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]
      
        ## todo: Try to load data/target
        train_data = diagrams[task_col <= 2]
        train_targets = task_col[task_col <= 2]
        train_targets[train_targets == 2] = 0

        test_data = diagrams[task_col > 2]
        test_targets = task_col[task_col > 2]
        test_targets[test_targets == 3] = 1
        test_targets[test_targets == 4] = 0

        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []
    

    model = LRModel()
    # model = LRFromScratch()
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
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)

