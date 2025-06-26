# Titanic Survival Prediction – Linear Regression Analysis

This project focuses on using linear regression to analyze and model the Titanic dataset. The objective is to predict the survival outcome of passengers using linear regression, evaluate the model using standard regression metrics, and visualize the relationship between selected features and predicted outcomes.

The analysis includes importing and preprocessing the dataset, splitting it into training and testing sets, fitting a linear regression model, evaluating it using MAE, MSE, and R², and visualizing the regression line to interpret feature influence.

## Import and Preprocess the Dataset

The dataset is first loaded using pandas. Irrelevant or non-numeric columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped. Missing values in `Age` are filled using the median, and missing values in `Embarked` are filled using the mode. Categorical variables like `Sex` and `Embarked` are label encoded to convert them into numerical form.

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Titanic.csv')

# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])
df['Embarked'] = label_enc.fit_transform(df['Embarked'])

# Check for nulls
print(df.isnull().sum())
```

## Split Data into Train-Test Sets

The dataset is divided into features (`X`) and target (`y`). The target is the `Survived` column, and all other columns are used as features. The data is split into training and testing sets using an 80-20 split.

```python
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Fit a Linear Regression Model using sklearn.linear_model

A linear regression model is created and trained using the training data. The model learns the relationship between the input features and the target survival variable.

```python
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
```

## Evaluate Model using MAE, MSE, R²

The model's performance is evaluated using regression metrics:
- Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
- Mean Squared Error (MSE): Penalizes larger errors more than MAE.
- R² Score: Indicates the proportion of variance in the target variable explained by the features.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
```

## Plot Regression Line and Interpret Coefficients

The regression coefficients are extracted to understand how each feature influences the predicted outcome. A regression line is plotted for one specific feature (`Fare`) to visualize how changes in that feature affect survival predictions, while all other features are held constant.

```python
# View feature coefficients
import pandas as pd

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lin_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_df)
```

```python
# Plot regression line for 'Fare'
import matplotlib.pyplot as plt
import seaborn as sns

fare_range = np.linspace(X['Fare'].min(), X['Fare'].max(), 300)
X_avg = pd.DataFrame(np.tile(X.mean().values, (300, 1)), columns=X.columns)
X_avg['Fare'] = fare_range

y_smooth = lin_model.predict(X_avg)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test['Fare'], y=y_test, label='Actual', alpha=0.6)
plt.plot(fare_range, y_smooth, color='red', label='Regression Line')
plt.xlabel('Fare')
plt.ylabel('Survival')
plt.title('Linear Regression: Fare vs Predicted Survival')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

The plot shows how predicted survival values change with `Fare`, assuming all other input features remain constant. Higher fare values are generally associated with a greater likelihood of survival, as indicated by the upward trend in the regression line.

This concludes the linear regression analysis of the Titanic dataset.
