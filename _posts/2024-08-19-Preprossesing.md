---
layout: post
title:  "Python Packages Help Automate Dataset Preprocessing"
date:   2024-08-03 9:31:29 +0900
categories: Update
---
### Python Packages Help Automate Dataset Preprocessing
There are several Python packages that can help automate dataset preprocessing, provide insights, and suggest improvements. Here are some popular ones:

1. **Pandas Profiling**:
   - Generates a detailed report with summaries and suggestions on a dataset, including missing values, correlations, outliers, and data type distributions.
   - Install it with `pip install pandas-profiling`.
   - Usage:
     ```python
     import pandas as pd
     from pandas_profiling import ProfileReport

     df = pd.read_csv("your_dataset.csv")
     profile = ProfileReport(df, title="Dataset Report")
     profile.to_notebook_iframe()  # Or save it with profile.to_file("report.html")
     ```

2. **Sweetviz**:
   - Similar to Pandas Profiling but focuses more on visualizations and comparisons, especially useful for comparing train and test datasets.
   - Install it with `pip install sweetviz`.
   - Usage:
     ```python
     import pandas as pd
     import sweetviz as sv

     df = pd.read_csv("your_dataset.csv")
     report = sv.analyze(df)
     report.show_html("sweetviz_report.html")
     ```

3. **AutoML Libraries with Preprocessing Capabilities**:
   - **Auto-Sklearn**, **TPOT**, and **H2O.ai AutoML** can handle not only preprocessing but also feature selection and model selection. They automate the entire ML pipeline, including data cleaning, feature engineering, and hyperparameter tuning.
   - Install with `pip install auto-sklearn`, `pip install tpot`, or `pip install h2o`.
   - Usage varies based on the library, but each has comprehensive documentation.

4. **DataPrep**:
   - Provides automated data cleaning and preprocessing, plus exploratory data analysis.
   - Install it with `pip install dataprep`.
   - Usage:
     ```python
     from dataprep.eda import create_report
     import pandas as pd

     df = pd.read_csv("your_dataset.csv")
     create_report(df)
     ```

5. **PyCaret**:
   - A low-code machine learning library that also offers data preprocessing, feature engineering, and model selection. It even has modules for data imputation, transformation, scaling, and encoding.
   - Install it with `pip install pycaret`.
   - Usage:
     ```python
     from pycaret.classification import setup

     setup(data=df, target="target_column")
     ```

Each of these tools provides a variety of automated insights and summaries, so you can choose the one that best fits your needs!


Of the packages mentioned, **PyCaret** and **TPOT** are designed to return a preprocessed dataset as part of their pipeline. Here’s how you can use each of them to get the preprocessed dataset:

### 1. **PyCaret**

PyCaret is a low-code machine learning library that not only preprocesses data but also prepares it for training. After setting up, it returns the preprocessed dataset and can show you what transformations were applied.

#### Example with PyCaret

```python
import pandas as pd
from pycaret.classification import setup, compare_models, get_config

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Setup environment for classification (change to 'pycaret.regression' for regression tasks)
s = setup(data=df, target="target_column", silent=True, session_id=123)

# Get the transformed training dataset
X_train = get_config('X_train')
y_train = get_config('y_train')

# Check the transformations applied (optional)
X_train.head()
```

Here, `get_config` gives access to the preprocessed training set `X_train` and the target labels `y_train`. PyCaret also performs automatic feature encoding, scaling, and outlier handling based on the setup.

### 2. **TPOT**

TPOT is an automated machine learning library that can optimize preprocessing steps and model selection. TPOT doesn’t explicitly return a preprocessed dataset, but it creates a preprocessing pipeline that you can apply to your data.

#### Example with TPOT

```python
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Split into features and target
X = df.drop("target_column", axis=1)
y = df["target_column"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

# Set up TPOT and fit to data
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Export the best pipeline
tpot.export("best_pipeline.py")

# Get the preprocessed data (optional)
pipeline = tpot.fitted_pipeline_
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)
```

In this example, `pipeline.transform(X_train)` returns the transformed dataset using TPOT’s chosen preprocessing pipeline. The exported Python file (`best_pipeline.py`) contains the code for the preprocessing and modeling steps, so you can see exactly what TPOT did to transform the data.

### 3. **DataPrep (EDA module)**

While DataPrep primarily focuses on creating reports and visualizing data, you can use its **cleaning** functionality from `DataPrep.Clean` to preprocess the dataset manually.

#### Example with DataPrep

```python
from dataprep.clean import clean_dates, clean_text
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Example of specific cleaning functions (text and date)
df_cleaned_dates = clean_dates(df, "date_column")  # Cleans date column
df_cleaned_text = clean_text(df, "text_column")    # Cleans text column

# To get the cleaned dataframe
df_preprocessed = df_cleaned_dates  # or combine them as needed
```

DataPrep won’t automatically preprocess the dataset but can be used for selective cleaning tasks on text, dates, or duplicates.

These examples demonstrate the most practical approaches in each package for retrieving preprocessed datasets!