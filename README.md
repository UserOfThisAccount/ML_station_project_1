# Product Demand Prediction with Machine Learning

## Project Overview
This project demonstrates how machine learning can be used to predict product demand during seasonal discount periods. By analyzing historical sales data, the goal is to find the optimal price point that will maximize both sales and profitability. The model is trained using past data on product pricing and units sold, and the **Decision Tree Regressor** algorithm is used to predict the units sold based on the prices.

---

## Features of the Model
1. **Base Price**: The initial price of the product before any discounts.
2. **Total Price**: The price at which the product is sold after applying any seasonal discounts.
3. **Units Sold**: The number of units sold at a given price.

The model uses `Base Price` and `Total Price` as features to predict `Units Sold`, which is our target variable.

---

## Objective
The objective of this project is to train a machine learning model to:
- Predict the number of units that will be sold based on the `Base Price` and `Total Price`.
- Help business owners set optimal prices during seasonal discounts to maximize sales while staying profitable.

---

## Dataset
The dataset used in this project contains the following columns:
- **ID**: Product ID (Unique identifier for each product)
- **Store ID**: ID of the store selling the product
- **Base Price**: Initial price of the product
- **Total Price**: Price at which the product was sold
- **Units Sold**: The number of units sold at the given price


---

## Installation Instructions

To set up this project and run it in your own environment, follow the steps below.

### 1. Installing Python

Make sure you have **Python 3.8 or higher** installed. You can check your version by running:
```bash
python --version
```

If you don’t have Python installed, download it from the [official website](https://www.python.org/downloads/).

### 2. Setting Up the Virtual Environment

To keep project dependencies isolated, it's recommended to create a virtual environment:
- Open your terminal or command prompt and navigate to your project folder.
- Create a virtual environment:
  ```bash
  python -m venv .venv
  ```

- Activate the virtual environment:
  - **On Windows**:
    ```bash
    .venv\Scripts\activate
    ```
  - **On macOS/Linux**:
    ```bash
    source .venv/bin/activate
    ```

### 3. Install the Required Libraries

Once the virtual environment is activated, install the necessary Python libraries using the **`requirements.txt`** file:

```bash
pip install -r requirements.txt
```

This will install the following dependencies:
- **`pandas`**: For data manipulation and analysis.
- **`scikit-learn`**: For building and training machine learning models.
- **`plotly`**: For data visualization.

### 4. Running the Code

Once the dependencies are installed:
- Ensure your dataset (e.g., `demand.csv`) is in the correct location.
- Run the code from a Jupyter notebook (`.ipynb`) or as a Python script.

### Example of How to Use the Model:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv('demand.csv')

# Define features and target
X = data[['Base Price', 'Total Price']]
y = data['Units Sold']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy (R² score): {accuracy}')
```

---

## Code Explanation

### 1. Data Loading and Preprocessing
The data is loaded using **Pandas** and cleaned to ensure there are no missing values. The relevant columns (`Base Price`, `Total Price`, and `Units Sold`) are selected as features for training the machine learning model.

### 2. Data Visualization
The relationships between `Base Price`, `Total Price`, and `Units Sold` are visualized using **Plotly's scatter plots** to help better understand how price changes affect demand.

### 3. Model Training
A **Decision Tree Regressor** from **scikit-learn** is used to train the model. The training data (`Base Price` and `Total Price`) is used to predict the target variable (`Units Sold`).

- The model is trained using `X_train` (features) and `y_train` (target).
- The performance of the model is evaluated using the **R² score** on the test data (`X_test` and `y_test`).

### 4. Model Evaluation
After training, the model’s accuracy is evaluated using the **R² score**. A score closer to 1.0 indicates a better fit. For example, an R² score of **0.95** means the model can explain 95% of the variance in the target variable.

### 5. Predictions
The model is tested on unseen data, and predictions are made for the test set. A plot is generated to compare the actual vs predicted sales figures.

---

## Model Evaluation

- **R² Score**: The **R² score** (coefficient of determination) measures the proportion of variance in the target variable (`Units Sold`) that is explained by the model. A higher R² score indicates a better model fit.

---

## Choice of Model
The **Decision Tree Regressor** was chosen because:
- It works well for regression tasks.
- It efficiently handles both continuous and categorical data.
- It handles non-linear relationships between features and the target variable.

---

## Conclusion

This project uses machine learning to predict product demand based on pricing, helping businesses set optimal prices during discount periods. By visualizing data and training a model, insights into the relationship between pricing and sales performance can be gained, improving decision-making.
