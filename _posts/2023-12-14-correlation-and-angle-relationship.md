---
layout: post
title: "Correlation and Angle Relationship"
date: 2023-12-14
categories: [math, data-science]
tags: [correlation, angle, math]
mathjax: true
---

# Correlation and Angle Relationship

The angle between two vectors (features) depends on their correlation coefficient, as it is directly related to the cosine of the angle between them:

$$
r = \cos(\theta)
$$

where $$ r $$ is the correlation coefficient ($$-1 \leq r \leq 1$$), and $$ \theta $$ is the angle between the two vectors.

---

## Relationship Between Correlation and Angle

### High Positive Correlation ($$ r \approx +1 $$):
- When $$ r \to 1 $$, $$ \cos(\theta) \to 1 $$, meaning $$ \theta \to 0^\circ $$.
- The vectors are nearly aligned in the same direction.
- **Example:** If $$ r = 0.9 $$, the angle is small:
  $$
  \theta = \cos^{-1}(0.9) \approx 25.84^\circ
  $$

### High Negative Correlation ($$ r \approx -1 $$):
- When $$ r \to -1 $$, $$ \cos(\theta) \to -1 $$, meaning $$ \theta \to 180^\circ $$.
- The vectors are nearly aligned in opposite directions.
- **Example:** If $$ r = -0.9 $$, the angle is:
  $$
  \theta = \cos^{-1}(-0.9) \approx 154.16^\circ
  $$

### Small Correlation ($$ r $$ is small):
- When $$ r $$ is small (e.g., $$ r \approx 0.1 $$ or $$ r \approx -0.1 $$), $$ \cos(\theta) $$ is close to $$ 0 $$, meaning $$ \theta $$ is close to $$ 90^\circ $$.
- The vectors are nearly orthogonal.
- **Example:** If $$ r = 0.1 $$, the angle is:
  $$
  \theta = \cos^{-1}(0.1) \approx 84.26^\circ
  $$

### Zero Correlation ($$ r = 0 $$):
- When $$ r = 0 $$, $$ \cos(\theta) = 0 $$, meaning $$ \theta = 90^\circ $$.
- The vectors are orthogonal or perpendicular.

---

## General Cases

| Correlation Coefficient ($r$)  | Angle ($\theta$)                | Relationship                             |
|--------------------------------|----------------------------------|------------------------------------------|
| $r = 1$                        | $0^\circ$                       | Perfect positive alignment               |
| $0 < r < 1$                    | $0^\circ < \theta < 90^\circ$   | Small acute angle (positive correlation) |
| $r = 0$                        | $90^\circ$                      | Orthogonal (no linear relationship)      |
| $-1 < r < 0$                   | $90^\circ < \theta < 180^\circ$ | Obtuse angle (negative correlation)      |
| $r = -1$                       | $180^\circ$                     | Perfect negative alignment               |


## Examples with Calculations

#### High Positive Correlation ($$ r = 0.8 $$):
$$
\theta = \cos^{-1}(0.8) \approx 36.87^\circ
$$

#### Low Positive Correlation ($$ r = 0.2 $$):
$$
\theta = \cos^{-1}(0.2) \approx 78.46^\circ
$$

#### High Negative Correlation ($$ r = -0.8 $$):
$$
\theta = \cos^{-1}(-0.8) \approx 143.13^\circ
$$

#### Small Negative Correlation ($$ r = -0.2 $$):
$$
\theta = \cos^{-1}(-0.2) \approx 101.54^\circ
$$

---

## Conclusion

**High correlation ($$ r \to 1 $$)** results in small angles ($$ \theta \to 0^\circ $$ or $$ \theta \to 180^\circ $$).

In **n-dimensional space**, the relationship between correlation, angle, and feature importance (and the decision to drop or keep features) becomes more complex but can still be understood through geometric and statistical principles. Here's a detailed explanation:

---

### **1. Correlation and Angle in n-Dimensional Space**
In an $$ n $$-dimensional feature space:
- **Correlation** measures the linear relationship between two features.
- The **angle** between two feature vectors is determined by their dot product, normalized by their magnitudes:
  $$
  \cos(\theta) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
  $$
  - A high correlation ($$ r \to 1 $$) means $$ \cos(\theta) \to 1 $$, so $$ \theta \to 0^\circ $$.
  - A low correlation ($$ r \to 0 $$) means $$ \cos(\theta) \to 0 $$, so $$ \theta \to 90^\circ $$.
  - A strong negative correlation ($$ r \to -1 $$) means $$ \cos(\theta) \to -1 $$, so $$ \theta \to 180^\circ $$.

In **n dimensions**, correlation can still be interpreted geometrically as the cosine of the angle between feature vectors, even though the data lives in higher dimensions.

---

### **2. Feature Importance and Redundancy**
- In $$ n $$-dimensional space, **redundant features** often have high correlation (small angles) and provide overlapping information.
- Key considerations for deciding whether to drop or keep features:
  1. **Correlation Threshold**: If two features have high correlation (e.g., $$ r > 0.9 $$), they are nearly collinear (small angle), and one may be dropped without losing much information.
  2. **Feature Importance**: Evaluate the importance of each feature using statistical or model-based methods (e.g., feature importance scores in tree-based models).
  3. **Dimensionality Reduction**: Instead of dropping features, use techniques like PCA to combine correlated features into uncorrelated components.

---

### **3. Impact of Correlation and Angle on Feature Selection**
#### **Case 1: High Correlation (Small Angle)**
- Highly correlated features are nearly collinear, meaning they span almost the same direction in the feature space.
- **Impact on Models:**
  - In linear models (e.g., regression), high correlation can cause **multicollinearity**, making it hard to estimate coefficients accurately.
  - Non-linear models (e.g., decision trees) are less affected but may still suffer from unnecessary complexity.
- **What to Do:**
  - Keep one feature based on domain knowledge or feature importance.
  - If unsure, use regularization (e.g., Lasso or Ridge) to select the most relevant feature.

#### **Case 2: Low Correlation (Large Angle, $$ \theta \approx 90^\circ $$)**
- Low correlation means the features are orthogonal or nearly independent.
- **Impact on Models:**
  - These features provide unique information and are generally beneficial for most machine learning models.
  - Removing one may lead to a loss of critical information.
- **What to Do:**
  - Retain both features unless one is irrelevant to the target variable (assessed via feature importance or statistical tests).

#### **Case 3: Negative Correlation ($$ \theta \approx 180^\circ $$)**
- Strong negative correlation indicates features are pointing in opposite directions but still carry similar information (linearly related).
- **Impact on Models:**
  - Similar to high positive correlation, negative correlation can introduce redundancy or multicollinearity in linear models.
- **What to Do:**
  - Consider dropping one feature or combining them (e.g., through PCA or weighted averages).

---

### **4. Dimensionality Reduction**
When dealing with high-dimensional datasets, **dimensionality reduction techniques** can help address correlated or redundant features:
- **Principal Component Analysis (PCA):**
  - Combines correlated features into uncorrelated components by projecting data onto orthogonal axes.
  - Helps reduce dimensionality while retaining most of the variance in the data.
- **Linear Discriminant Analysis (LDA):**
  - Focuses on maximizing class separability and can be used in classification tasks.

---

### **5. Feature Importance and Decision to Drop/Keep**
Use model-based methods or statistical techniques to evaluate feature importance:
- **Tree-based Models (e.g., Random Forest, Gradient Boosting):**
  - Provide feature importance scores based on their contribution to splits in decision trees.
- **Regularization (Lasso, Ridge):**
  - Penalizes less important features and reduces their coefficients toward zero.
- **SHAP (SHapley Additive exPlanations):**
  - Provides interpretability for feature contributions to predictions.

#### **Guidelines for Dropping Features:**
1. **Drop features if:**
   - High correlation exists ($$ r > 0.9 $$).
   - Feature importance score is low.
   - Domain knowledge suggests irrelevance.
2. **Keep features if:**
   - Low correlation ($$ r \approx 0 $$) indicates unique information.
   - Feature importance is high.

---

### **6. Summary Table**

| **Correlation** | **Angle ($$ \theta $$)**      | **Feature Relationship**         | **Action**                                      |
|------------------|-------------------------------|-----------------------------------|------------------------------------------------|
| $$ r = 1 $$      | $$ 0^\circ $$                | Perfectly aligned, redundant     | Drop one feature.                              |
| $$ 0.8 \leq r < 1 $$ | $$ 0^\circ < \theta < 45^\circ $$ | Highly correlated               | Consider dropping one (or combining).          |
| $$ 0 < r < 0.8 $$ | $$ 45^\circ < \theta < 90^\circ $$ | Moderately correlated           | Keep both, unless feature importance is low.   |
| $$ r = 0 $$      | $$ 90^\circ $$               | Orthogonal (independent)         | Retain both features.                          |
| $$ -0.8 < r < 0 $$ | $$ 90^\circ < \theta < 135^\circ $$ | Moderately negatively correlated | Evaluate importance; combine if redundant.     |
| $$ r = -1 $$     | $$ 180^\circ $$              | Perfectly negatively aligned     | Drop one feature.                              |

---

### **Conclusion**
- **Correlation and angle** reveal redundancy and independence in feature space.
- **Dimensionality reduction** can address correlation without dropping features.
- Use **domain knowledge** and **feature importance methods** to decide whether to drop or keep features.
