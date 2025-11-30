# Sales Forecasting Web Application - Deployment

This web application allows users to predict future sales using a pre-trained neural network model. The application is built with **Streamlit** for the user interface and **TensorFlow/Keras** for the predictive model.

## Deployment Link

The application is live and can be accessed at:

[Sales Forecasting Web App](https://sales-forecasting.up.railway.app/#complete-results-table)

## Features

The deployed application provides the following functionality:

- Load a pre-trained neural network model.
- Optionally load feature scalers to normalize input data.
- Upload CSV files containing store and product features.
- Generate sales predictions for future dates.
- Download the predicted results as a CSV file.
- Track prediction history and session statistics.

## Usage Instructions

1. Open the web app using the deployment link above.
2. Load the model from the sidebar.
3. Optionally load feature scalers for better prediction accuracy.
4. Upload your CSV file containing the required features.
5. Configure prediction options (e.g., apply scaling if needed).
6. Click **Predict Sales** to generate forecasts.
7. Download the results as a CSV file.
8. Check your session history for past predictions.

## Notes

- Ensure your CSV file contains all required features in the correct order.
- The app automatically handles feature scaling if scalers are loaded.
- Predictions are displayed immediately, along with summary metrics.
