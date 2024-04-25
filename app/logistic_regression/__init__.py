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
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """

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
        """ Evaluate the model

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
        class_report = classification_report(y_test, y_pred)
        return roc_auc, conf_matrix, class_report
    
    def plot_roc_curve(self, y_test: pd.Series, y_pred_prob: pd.Series) -> None:
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
        
    def run(self, filename: str) -> None:
        df = self.load_data(filename)
        X_train, X_test, y_train, y_test = self.split_data(df)
        model = self.train_model(X_train, y_train)
        roc_auc, conf_matrix, class_report = self.evaluate_model(model, X_test, y_test)
        self.plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        self.log.info(f"ROC-AUC Score: {roc_auc}")
        self.log.info(f"Confusion Matrix:\n{conf_matrix}")
        self.log.info(f"Classification Report:\n{class_report}")
        self.log.info("Model trained, evaluated, and ROC curve plotted successfully.")