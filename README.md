# DS675-RiskBasedAuth

## Kaggle API Key Instructions

- Create or login to your Kaggle account.
- Go to your Account settings.
- Scroll to the API section and click the "Create New API Token" button.
- Kaggle will generate a JSON file named kaggle.json and prompt you to save the file to your computer.
- Download this file and save it to a secure location on your computer

Create a `.env` file in the root directory with the following content and replace `/path/to/kaggle.json` with the path to the kaggle.json file you downloaded.

## .env

```ini
# .env
KAGGLE_CONFIG_PATH=/path/to/kaggle.json
```

## Installation/Setup

- Start by cloning the repository

```bash
git clone git@github.com:Carlomos7/DS675-RiskBasedAuth.git
cd DS675-RiskBasedAuth
```

- Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

- Install the required packages

```bash
pip install -r requirements.txt
```

## Running the code

- To run the code, please do so in the following order:

### Logistic Regression

```bash
# this will pull the kaggle dataset into data/kaggle_data/
# And it runs the logistic_regression logic - @Carlomos7
python app/main.py
```

### Gradient Boosting

```bash
python app/gradient_boosting.py
```

### Random Forest

```bash
python app/random_forest_regressor.py
```
