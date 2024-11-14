---
layout: post
title:  "Understanding Dimensionality Reduction for High-Dimensional Data Visualization"
date:   2024-11-03 9:31:29 +0900
categories: Update
---
### Understanding Dimensionality Reduction for High-Dimensional Data Visualization

In this section, we will cover:

1. **The Importance of Dimensionality Reduction**: 

Discuss why dimensionality reduction is essential for visualizing and analyzing complex, high-dimensional data.

2. **Techniques Overview**: 

Provide a brief explanation of PCA, LDA, t-SNE, and UMAP, highlighting their strengths and best-use cases.

3. **Choosing the Right Technique**: 

Guide users on selecting the best method depending on the dataset and objectives, perhaps with visual examples.

4. **Applications and Examples**: 

Show specific scenarios (like image data, text, or clustering) where these techniques are applied effectively.

5. **Limitations and Trade-Offs**: 

Discuss common challenges, such as interpretability, parameter tuning, and computational cost, to help users understand when and how to apply these methods effectively. 

This would give readers both an informative and practical understanding of dimensionality reduction in data science.
Analyze and visualize the statistical properties and distributions of a dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class AbstractPreprocessor(ABC):
    def __init__(self, data_path, file_type='csv', irrelevant_columns=[]):
        """
        Initializes the preprocessor with data from a specified file.
        :param data_path: Path to the dataset (csv or json).
        :param file_type: Format of the dataset, either 'csv' or 'json'.
        :param irrelevant_columns: List of column names to drop.
        """
        self.data_path = data_path
        self.file_type = file_type
        self.irrelevant_columns = irrelevant_columns
        self.df = self.load_data()
        
    def load_data(self):
        """
        Loads the dataset based on the file type (csv or json).
        """
        if self.file_type == 'csv':
            return pd.read_csv(self.data_path)
        elif self.file_type == 'json':
            return pd.read_json(self.data_path)
        else:
            raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")

    @abstractmethod
    def visualize_before(self):
        pass

    @abstractmethod
    def visualize_after(self):
        pass

    def remove_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print(f"Removing {duplicates} duplicate rows.")
        self.df = self.df.drop_duplicates()

    def remove_irrelevant(self):
        print(f"Removing irrelevant columns: {self.irrelevant_columns}")
        self.df = self.df.drop(columns=self.irrelevant_columns, errors='ignore')

    def identify_null_values(self):
        print("Identifying null values and other missing indicators...")
        null_counts = self.df.isnull().sum()
        blank_counts = (self.df == "").sum()
        print("Null values per column:\n", null_counts[null_counts > 0])
        print("Blank values per column:\n", blank_counts[blank_counts > 0])

    def identify_extreme_values(self):
        print("Identifying columns with extreme values and unique values...")
        for column in self.df.select_dtypes(include=[np.number]):
            min_value = self.df[column].min()
            max_value = self.df[column].max()
            zero_count = (self.df[column] == 0).sum()
            unique_count = self.df[column].nunique()
            print(f"Column '{column}': min={min_value}, max={max_value}, zero_count={zero_count}, unique_count={unique_count}")

    def calculate_statistics(self, column):
        """
        Calculates and prints key statistics for a given column.
        """
        stats = {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'std': self.df[column].std(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            '25th_percentile': self.df[column].quantile(0.25),
            '50th_percentile': self.df[column].quantile(0.50),
            '75th_percentile': self.df[column].quantile(0.75),
            'skew': self.df[column].skew()
        }
        print(f"Statistics for '{column}':")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")
        return stats

    def analyze_distributions(self, columns=None, target_column=None):
        """
        Analyzes distributions of specified columns and visualizes them using boxplots, density plots, and histograms.
        :param columns: List of columns to analyze. If None, analyzes all numeric columns.
        :param target_column: Optional target column for class-based visualization.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            print(f"\nAnalyzing distribution for '{column}':")
            self.calculate_statistics(column)

            # Visualization
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            if target_column and target_column in self.df.columns:
                sns.boxplot(data=self.df, x=target_column, y=column, ax=axs[0])
                sns.kdeplot(data=self.df, x=column, hue=target_column, ax=axs[1])
                sns.histplot(data=self.df, x=column, hue=target_column, kde=True, ax=axs[2], bins=30)
            else:
                sns.boxplot(x=self.df[column], ax=axs[0])
                sns.kdeplot(x=self.df[column], ax=axs[1])
                sns.histplot(x=self.df[column], kde=True, ax=axs[2], bins=30)

            axs[0].set_title(f"Boxplot of {column}")
            axs[1].set_title(f"Density Plot of {column}")
            axs[2].set_title(f"Histogram of {column}")
            plt.show()

    def preprocess(self, outlier_columns=[], missing_strategy='mean', outlier_strategy='cap'):
        """
        Runs the entire preprocessing pipeline: duplicates, irrelevant columns, null handling, extreme values, outliers, and missing values.
        :param outlier_columns: List of columns to check for outliers.
        :param missing_strategy: Strategy to handle missing values.
        :param outlier_strategy: Strategy to handle outliers.
        """
        self.remove_duplicates()
        self.remove_irrelevant()
        self.identify_null_values()
        self.identify_extreme_values()
        
        for column in outlier_columns:
            self.handle_outliers(column, strategy=outlier_strategy)
        
        self.handle_missing_values(strategy=missing_strategy)
        self.statistical_analysis()
        self.check_correlations()

    def plot_distributions(self, columns, title='Distribution of Features'):
        plt.figure(figsize=(12, 8))
        for col in columns:
            sns.histplot(self.df[col], kde=True, label=col)
        plt.title(title)
        plt.legend()
        plt.show()

# Example Concrete Implementation
class MyPreprocessor(AbstractPreprocessor):
    def visualize_before(self):
        print("Visualizing data before preprocessing...")
        self.plot_distributions(self.df.columns, title='Before Preprocessing')

    def visualize_after(self):
        print("Visualizing data after preprocessing...")
        self.plot_distributions(self.df.columns, title='After Preprocessing')

# Usage Example
# data_path = "your_dataset.csv"
# irrelevant_columns = ['irrelevant_feature']
# preprocessor = MyPreprocessor(data_path, file_type='csv', irrelevant_columns=irrelevant_columns)
# preprocessor.visualize_before()
# preprocessor.analyze_distributions(columns=['your_numeric_column'], target_column='target_class')
# preprocessor.visualize_after()
```

### Explanation of `analyze_distributions`

1. **Statistics Calculation**:
   - The `calculate_statistics` method computes key statistics for each specified column: mean, median, standard deviation, minimum, maximum, percentiles, and skewness.
   - These statistics help identify the central tendency, spread, and skewness, which guide the choice of transformations (e.g., log or square root) and scaling methods.

2. **Visualization**:
   - For each column, the method produces three plots:
     - **Boxplot**: Highlights outliers, median, and interquartile range (IQR), ideal for spotting distribution spread and skewness.
     - **Density Plot**: Shows the continuous shape of the distribution, useful for visualizing skewness.
     - **Histogram**: Provides a bar representation of value frequencies, ideal for spotting skewness and data range.
   - If a target column is provided (for multi-class analysis), the visualizations show feature distributions across classes, which helps understand feature-target relationships.

3. **Optional Target-Based Plotting**:
   - When `target_column` is specified, the method visualizes each feature's distribution per class. This helps identify which features are most distinct or predictive for different classes.

This updated `analyze_distributions` method will give you a comprehensive view of each feature’s distribution, helping you make informed decisions about scaling, transformation, or outlier handling. Let me know if you'd like further customization!