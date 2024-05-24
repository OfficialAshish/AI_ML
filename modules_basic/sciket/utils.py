


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class ModelEvaluator:
    def __init__(self, model, X_train, y_train, X_test, y_test, is_classifier=True):
        """
        
        Example usage:
        Assuming `model` is your pre-trained model, and X_train, X_test, y_train, y_test are your datasets
        
        For Classifier
            evaluator = ModelEvaluator(model, X_train, y_train, X_test, y_test, is_classifier=True)
            evaluator.evaluate()
        
        For Regressor
            evaluator = ModelEvaluator(model, X_train, y_train, X_test, y_test, is_classifier=False)
            evaluator.evaluate()
    
        Usage:
            model_evaluator = ModelEvaluator(rfc, X_train, y_train, X_test, y_test)
            model_evaluator.cross_validate_accuracy()
            model_evaluator.plot_roc_curve()
            model_evaluator.plot_confusion_matrix()
            model_evaluator.plot_classification_report()
            model_evaluator.display_confusion_matrix()
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_classifier = is_classifier
        
        # Fit the model if not already fitted
        if not hasattr(model, "fit"):
            raise ValueError("Provided model instance does not have a fit method.")
        
        self.fitted_model = clone(model).fit(X_train, y_train)
        self.y_preds = self.fitted_model.predict(X_test)
        
        if is_classifier:
            self.y_pred_probs = self.fitted_model.predict_proba(X_test)[:, 1] if hasattr(self.fitted_model, 'predict_proba') else None

    def cross_validate_accuracy(self, cv=5):
        score_on_cv = cross_val_score(self.fitted_model, self.X_train, self.y_train, cv=cv)
        mean_accuracy = np.mean(score_on_cv) * 100
        print("Scores on different splits of datasets:\n", score_on_cv)
        print(f"Classifier Cross-Validated Accuracy: {mean_accuracy:.2f}%\n") 

    def plot_roc_curve(self):
        if self.is_classifier and self.y_pred_probs is not None:
            RocCurveDisplay.from_estimator(self.fitted_model, self.X_test, self.y_test)
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.show()
        else:
            print("ROC curve is not applicable for regressors or classifiers without probability estimates.")

    def plot_classification_report(self):
        if self.is_classifier:
            c_rep = classification_report(self.y_test, self.y_preds, output_dict=True)
            report_df = pd.DataFrame(c_rep).transpose()
            print("Classification Report:\n", report_df)
        else:
            print("Classification report is not applicable for regressors.")

    def display_confusion_matrix(self):
        if self.is_classifier:
            disp = ConfusionMatrixDisplay.from_estimator(self.fitted_model, self.X_test, self.y_test)
            plt.title("Confusion Matrix")
            disp.plot()
            plt.show()
        else:
            print("Confusion matrix is not applicable for regressors.")

    def evaluate_regressor(self):
        if not self.is_classifier:
            mse = mean_squared_error(self.y_test, self.y_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, self.y_preds)
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Root Mean Squared Error: {rmse:.2f}")
            print(f"R^2 Score: {r2:.2f}")
        else:
            print("Regressor evaluation metrics are not applicable for classifiers.")

    def evaluate(self, cv=5):
        # Perform cross-validation accuracy
        self.cross_validate_accuracy(cv=cv)
        
        if self.is_classifier:
            # Perform classifier-specific evaluations
            self.plot_roc_curve()
            self.plot_classification_report()
            self.display_confusion_matrix()
        else:
            # Perform regressor-specific evaluations
            self.evaluate_regressor()

# Example usage:
# Assuming `model` is your pre-trained model, and X_train, X_test, y_train, y_test are your datasets

# For Classifier
# evaluator = ModelEvaluator(model, X_train, y_train, X_test, y_test, is_classifier=True)
# evaluator.evaluate()

# For Regressor
# evaluator = ModelEvaluator(model, X_train, y_train, X_test, y_test, is_classifier=False)
# evaluator.evaluate()

























# """
# Classification model evaluation metrics
# 1. Accuracy
# 2. Area under ROC curve
# 3. Confusion matrix
# 4. Classification report
# """
# from sklearn.model_selection import cross_val_score
# # Model is trained on different versions of training data, and evaluated on different versions of test data.

# score_on_cv = cross_val_score(rfc, X, y, cv = 5)
# print("Score on different version(split) of datasets\n: ", score_on_cv)
# print(f" Classifier Cross—Validated Accuracy: {np.mean(score_on_cv) *100:.2f}%\n")

# # ROC curves (AUC) are a comparison of a model's true postive rate (tpr) versus a models false positive rate (fpr) .

# # True positive = model predicts 1 when truth is 1
# # False positive = model predicts 1 when truth is 0
# # True negative = model predicts 0 when truth is 0
# # False negative = model predicts 0 when truth is 1

# y_probs = rfc.predict_proba(X_test)
# from sklearn.metrics import roc_curve
# # Caculate fpr, tpr and thresholds
# y_probs_positive = y_probs[:,1]
# fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

# # Plotting ROC curve
# import matplotlib.pyplot as plt

# from sklearn.metrics import roc_auc_score

# def plot_roc_curve(fpr, tpr, ax=None):
#     """
#     Plots a ROC curve given the false positive rate (fpr)
#     and true positive rate (tpr) of a model.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(7, 7))
    
#     ax.plot(fpr, tpr, color="orange", label="ROC")
#     ax.plot([0, 1], [0, 1], color="green", label="Guessing", linestyle="--")
    
#     ax.set_xlabel("False positive rate (fpr)")
#     ax.set_ylabel("True positive rate (tpr)")
#     ax.set_title("Receiver Operating Characteristic (ROC) Curve")
#     ax.legend()
#     print(roc_auc_score(y_test, y_probs_positive))
    
#     return ax

# # or this 
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator( rfc, X_test, y_test)
# plt.show()

# # Confusion Matrix
# # A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.

# # In short way, we can directly plot/visualize this
# pd.crosstab(y_test, y_preds, rownames = [ "Actual",], colnames = ["Predicted",])

# # Another way
# # Make our confusion matrix more visual with Seaborn 's heatmap()
# from sklearn.metrics import confusion_matrix
# conf_mat = confusion_matrix(y_test, y_preds)
# print("Conf. Matrix:",conf_mat)

# import seaborn as sns
# # Set the font scale
# sns.set(font_scale = 1.5)
# # sns.heatmap(conf_mat);
# # y_test.value_counts()

# def plot_confusion_mat(conf_mat):
#     """Plots confusion matrix using seaborn's heatmap"""
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax = sns.heatmap(conf_mat,
#                     annot=True,
#                     cbar=True)
#     ax.set(
#         xlabel="Predicted label", ylabel="True label",
#         title="Confusion Matrix"
#     )

# plot_confusion_mat(conf_mat)

# # or with this
# from sklearn.metrics import  ConfusionMatrixDisplay
# disp1 = ConfusionMatrixDisplay.from_estimator(rfc, X_test, y_test)


# from sklearn.metrics import classification_report
# c_rep = classification_report(y_test, y_preds)
# print(c_rep)

# def plot_classification_report(y_test, y_preds):
#     """
#     Plots a bar chart for precision, recall, and f1-score for each class in the classification report.

#     Parameters:
#     - y_test (array-like): True labels.
#     - y_preds (array-like): Predicted labels.
#     """
#     report_data = classification_report(y_test, y_preds, output_dict=True)
#     metrics = ['precision', 'recall', 'f1-score']
#     class_names = [cls for cls in report_data.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
#     class_data = {cls: [report_data[cls][metric] for metric in metrics] for cls in class_names}

#     fig, ax = plt.subplots(figsize=(7, 10))

#     width = 0.2

#     # Class-wise metrics
#     for i, cls in enumerate(class_names):
#         ax.bar(np.arange(len(metrics)) + i * width, class_data[cls], width=width,  label=f'Class {cls} (support: {int(report_data[cls]["support"])})')
#         # ax.bar(np.arange(len(metrics)) + i * width, class_data[cls], width=width, label='Class ' + cls)

#      # # Adding support values as annotations
#     # for i, cls in enumerate(class_names):
#     #     for j, metric in enumerate(metrics):
#     #         ax.text(j + i * width, class_data[cls][j], f'{int(report_data[cls]["support"])}', ha='center', va='bottom')

#     ax.set_xticks(np.arange(len(metrics)) + width)
#     ax.set_xticklabels(metrics)
#     ax.set_ylabel('Score')
#     ax.set_title('Classification Report')
#     ax.legend(loc='best', fontsize="x-small" , bbox_to_anchor=(0.5, -0.1))

    
#     plt.show()

