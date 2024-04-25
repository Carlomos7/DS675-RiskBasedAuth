""" This is the logistic regression module.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path
from config import get_settings
from log_config import get_logger
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    """Logistic regression model class for the RBA model"""
    
    def __init__(self):
        self.log = get_logger(__name__)
        self.config = get_settings()
        self.sample_data_directory = Path(self.config.SAMPLE_DATA_DIRECTORY)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load the preprocessed dataset

        Args:
            filename (str): The filename of the preprocessed dataset

        Returns:
            pd.DataFrame: The loaded preprocessed dataset
        """
        self.log.info("Loading preprocessed dataset...")
        self.df = pd.read_csv(filename)
        self.log.info("Dataset loaded successfully.")
        return self.df
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split the dataset into training and testing sets
        
        Args:
            df (pd.DataFrame): The DataFrame to split
            
        Returns:
            tuple: The training and testing sets
        """
        X = df.drop('Is Account Takeover', axis=1)
        y = df['Is Account Takeover']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """ Train the logistic regression model

        Args:
            X_train (pd.DataFrame): DataFrame of features
            y_train (pd.Series): Series of target values

        Returns:
            LogisticRegression: Trained logistic regression model
        """
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
        """ Evaluate the model by calculating the ROC-AUC (ROC - Area Under Curve) score, confusion matrix, and classification report (precision, recall, f1-score, support)
        Referenced: https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        
        Args:
            model (LogisticRegression): Model to evaluate
            X_test (pd.DataFrame): DataFrame of features
            y_test (pd.Series): Series of target values

        Returns:
            tuple: ROC-AUC score, confusion matrix, classification report
        """
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Referenced: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        class_report = classification_report(y_test, y_pred, zero_division=1)
        return roc_auc, conf_matrix, class_report
    
    def plot_roc_curve(self, y_test: pd.Series, y_pred_prob: pd.Series) -> None:
        """ Plot the ROC curve
        Referenced: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
        
        Args:
            y_test (pd.Series): The true values
            y_pred_prob (pd.Series): The predicted probabilities 
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        self.log.info("Plotting the ROC curve...")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        image_save_path = self.sample_data_directory / 'lr_roc_curve.png'
        self.log.info("Saving plot ...")
        plt.savefig(image_save_path)
        self.log.info(f"ROC curve saved to {image_save_path}")
    
    def get_risk_factors(self, x, y) -> pd.DataFrame:
        """ Get the risk factors for the model
        
        Args:
            x (list): The predicted values
            y (list): The true values
            
        Returns:
            pd.DataFrame: DataFrame containing predicted values, true values, and risk factors
        """
        risk_factors = []
        for pred, true in zip(x, y):
            if pred == 1 and true == 1:
                risk_factors.append("True Positive")
            elif pred == 1 and true == 0:
                risk_factors.append("False Positive")
            elif pred == 0 and true == 1:
                risk_factors.append("False Negative")
            else:
                risk_factors.append("True Negative")
        
        # Create a DataFrame with predicted values, true values, and risk factors
        risk_factors_df = pd.DataFrame({
            'Predicted': x,
            'True': y,
            'Risk Factor': risk_factors
        })
        
        return risk_factors_df
    
    def plot_risk_factor_distribution(self, df: pd.DataFrame) -> None:
        """ Plot the distribution of risk factors

        Args:
            df (pd.DataFrame): The DataFrame containing the risk factors
        """
        df['Risk Factor'].value_counts().plot(kind='bar',figsize=(10,6))
        plt.title('Risk Factor Distribution')
        plt.xlabel('Risk Factor')
        plt.ylabel('Count')
        plt.show()
        image_save_path = self.sample_data_directory / 'lr_risk_factor_distribution.png'
        self.log.info("Saving plot ...")
        plt.savefig(image_save_path)
        self.log.info(f"ROC curve saved to {image_save_path}")
        
        
    def export_data(self, df: pd.DataFrame, filename: str) -> None:
        """Export preprocessed data to CSV

        Args:
            df (pd.DataFrame): The DataFrame to export
            filename (str): The filename to export the preprocessed data
        """
        self.log.info("Exporting data to csv...")
        df.to_csv(filename, index=False)
        self.log.info(f"Data exported to {filename}")

    def run(self, filename: str) -> None:
        df = self.load_data(filename)
        X_train, X_test, y_train, y_test = self.split_data(df)
        model = self.train_model(X_train, y_train)
        roc_auc, conf_matrix, class_report = self.evaluate_model(model, X_test, y_test)
        risk_factors = self.get_risk_factors(model.predict(X_test), y_test)
        self.export_data(risk_factors, self.sample_data_directory / 'risk_factors.csv')
        self.plot_risk_factor_distribution(risk_factors)
        self.plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        self.log.info(f"ROC-AUC Score: {roc_auc}")
        self.log.info(f"Confusion Matrix:\n{conf_matrix}")
        self.log.info(f"Classification Report:\n{class_report}")
        self.log.info("Model trained, evaluated, and ROC curve plotted successfully.")