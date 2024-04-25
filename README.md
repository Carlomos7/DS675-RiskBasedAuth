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
