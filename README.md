# Iris Dataset Classification and Visualization

This project demonstrates a complete machine-learning workflow using the classic **Iris dataset**. It includes data loading, visualization, preprocessing, model training with logistic regression, hyperparameter tuning, and performance evaluation with multiple metrics, including ROC curves.

## 📁 Dataset

The dataset used is `iris.csv`, located in the `2- Dataset/` directory. It includes measurements of iris flowers from three species:
- Setosa
- Versicolor
- Virginica

Each observation contains the following features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## 📊 Exploratory Data Analysis (EDA)

The notebook performs visual analysis using **Seaborn** to understand the data distribution and relationships:
- **Scatterplots** of `petal_length` vs `petal_width` and `sepal_length` vs `sepal_width`, colored by species.
- **Correlation heatmap** of numerical features to assess linear relationships.

## ⚙️ Preprocessing

- The target variable (`species`) is separated from the feature set.
- Data is split into training and testing sets (75/25 split).
- Features are standardized using `StandardScaler`.

## 🧠 Model Training and Hyperparameter Tuning

A **Logistic Regression** model is trained with extensive hyperparameter tuning using `GridSearchCV`. The parameters tuned include:
- `penalty`: `l1`, `l2`, `elasticnet`
- `C`: Regularization strength (logarithmic scale)
- `l1_ratio`: Used with `elasticnet`

> Solver used: `saga` with `max_iter=5000` to ensure convergence.

## ✅ Model Evaluation

The following evaluation techniques are applied:
- **Accuracy score**
- **Confusion matrix** (with visual display)
- **Classification report** (precision, recall, F1-score for each class)

## 📈 ROC Curve Analysis

The model’s performance is further evaluated using:
- **Multiclass ROC curves** (one for each class)
- **Macro-average ROC curve**, representing average performance across all classes

> Classes are binarized for ROC curve calculation using `label_binarize` from `sklearn.preprocessing`.

## 🛠 Libraries Used

- `pandas`, `numpy`
- `matplotlib.pyplot`, `seaborn`
- `scikit-learn` modules: `model_selection`, `linear_model`, `metrics`, `preprocessing`

## 📌 How to Run

1. Ensure you have Python 3.x and the required libraries installed.
2. Place the `iris.csv` file in the `2- Dataset/` folder.
3. Open the notebook and run all cells in order to follow the full workflow.
