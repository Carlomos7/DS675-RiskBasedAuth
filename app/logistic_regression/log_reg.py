import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path
from config import get_settings
from log_config import get_logger
import matplotlib.pyplot as plt

def log_reg():
    # Initialize logging and configuration
    log = get_logger('log_reg')
    config = get_settings()

    # Load the preprocessed dataset
    log.info("Loading preprocessed dataset...")
    sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
    stratified_sample_filename = sample_data_directory / f"pre-processed_subset_{config.DATA_FILE}"
    df = pd.read_csv(stratified_sample_filename)

    # Splitting dataset into features and target
    X = df.drop('Is Account Takeover', axis=1)
    y = df['Is Account Takeover']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning
    #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight': ['balanced', None]}
    #log.info("Hyperparameter tuning...")
    #grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='roc_auc')
    #grid_search.fit(X_train, y_train)

    # Training logistic regression model with best parameters
    log.info("Training logistic regression model with best parameters...")
    model = LogisticRegression() #max_iter=1000, **grid_search.best_params_
    model.fit(X_train, y_train)

    # Evaluate the model
    log.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    log.info(f"ROC-AUC Score: {roc_auc}")
    log.info(f"Confusion Matrix:\n{conf_matrix}")
    log.info(f"Classification Report:\n{class_report}")

    # Plotting the ROC curve
    log.info("Plotting the ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
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

    # Final output
    log.info("Final model trained, evaluated, and ROC curve plotted successfully.")

log_reg()