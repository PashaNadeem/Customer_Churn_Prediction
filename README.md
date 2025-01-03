# Customer Churn Prediction

## Introduction
Customer churn is a critical metric for businesses to monitor and predict, as it directly impacts revenue and growth. This project leverages machine learning techniques to predict customer churn using a dataset containing customer demographics, subscription details, and behavioral patterns.

## Project Objectives
1. Analyze customer data to identify factors influencing churn.
2. Build and evaluate machine learning models for predicting customer churn.
3. Provide actionable insights to improve customer retention strategies.

## Dataset Overview
The dataset includes the following columns:

- **CustomerID**: Unique identifier for each customer.
- **Age**: Customer's age.
- **Gender**: Customer's gender (Male, Female).
- **Tenure**: Number of months the customer has been with the service.
- **Usage Frequency**: Frequency of service usage.
- **Support Calls**: Number of customer support calls made.
- **Payment Delay**: Days of delay in payments.
- **Subscription Type**: Type of subscription (Basic, Standard, Premium).
- **Contract Length**: Contract duration (Monthly, Annually, Quarterly).
- **Total Spend**: Total amount spent by the customer.
- **Last Interaction**: Days since the last interaction.
- **Churn**: Target variable (1 for churned, 0 for retained).

## Implementation Steps

### Step 1: Import Libraries
Imported essential Python libraries for data manipulation, visualization, and machine learning, including:
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for preprocessing, modeling, and evaluation

### Step 2: Load and Inspect the Data
- Loaded the dataset and inspected its structure.
- Checked for missing values and column data types.

### Step 3: Preprocess the Data
- **Handled Missing Values**: Ensured no missing data by performing imputation or removal.
- **Encoded Categorical Variables**: Converted `Gender`, `Subscription Type`, and `Contract Length` into numerical representations using OneHotEncoder.
- **Scaled Numerical Features**: Applied `StandardScaler` to normalize numerical features like `Age`, `Tenure`, and `Total Spend`.

### Step 4: Exploratory Data Analysis (EDA)
- Visualized the distribution of numerical variables and relationships between features using:
  - Histograms, box plots, and pair plots.
  - Heatmap for correlation analysis to identify highly correlated features.

### Step 5: Model Building
#### Models Implemented:
1. **Random Forest Classifier**
2. **Logistic Regression**

### Step 6: Model Evaluation
- Evaluated the models using:
  - **Accuracy**
  - **Precision, Recall, F1-Score** (from classification reports)
  - **ROC-AUC Score** for overall performance.

### Step 7: Hyperparameter Tuning with GridSearchCV
- Applied **GridSearchCV** to tune hyperparameters for the Random Forest and Logistic Regression models.
- Optimized parameters for better accuracy and performance.

### Results
- **Logistic Regression** achieved an accuracy of ~90% (ROC-AUC Score).
- **Random Forest Classifier** achieved an accuracy of ~99.9%.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn

## Results and Insights
- Identified key factors influencing churn, such as payment delays and usage frequency.
- The supervised approach yielded actionable predictions, while clustering provided additional segmentation insights.

## Future Work
1. Experiment with additional algorithms like XGBoost or Gradient Boosting.
2. Incorporate external data, such as customer feedback or market trends, for improved predictions.
3. Implement hyperparameter tuning to optimize model performance.

## Instructions for Use
1. Clone the repository:
   ```bash
   git clone https://github.com/PashaNadeem/customer-churn-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-churn-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook to execute the analysis step by step.

## Acknowledgements
Special thanks to the open-source community and datasets used for this project.

