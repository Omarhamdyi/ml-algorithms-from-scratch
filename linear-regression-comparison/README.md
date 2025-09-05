# Linear Regression: Gradient Descent vs Normal Equation vs Scikit-learn
<p align="center">
  <img  width="70%" src="linear-regression-comparison.jpg" alt="linear regression comparison">

## 📑 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [How It Works](#-how-it-works)
- [Implemented Methods](#-implemented-methods)
  - [Gradient Descent](#1-gradient-descent)
  - [Normal Equation](#2-normal-equation)
  - [Scikit-learn](#3-scikit-learn-implementation)
- [Results](#-results)
- [Time Complexity Analysis](#-time-complexity-analysis)
- [Key Takeaways](#-key-takeaways)
- [Future Work](#-future-work)
- [Author](#-author)

## 📌 Overview
This project implements and compares *three approaches* to Linear Regression:
1. *Gradient Descent implementation (from scratch)*
2. *Normal Equation implementation (from scratch)*
3. *Scikit-learn implementation*

The goal is to understand the differences between these methods in terms of:
- Implementation complexity
- Computation (time & resources)
- Accuracy (Mean Squared Error)

---

## 📂 Project Structure

src/
├── gradient_descent.py     # Linear Regression using Gradient Descent

  ├── normal_equation.py      # Linear Regression using Normal Equation
  
  ├── sklearn_impl.py         # Linear Regression using scikit-learn
  
  ├── main.py                 # Calls all implementations & stores results
  
  └── visualization.py        # Plots results for comparison



---

## 📦 Dependencies
The project requires the following Python libraries:
bash
numpy
pandas
matplotlib
scikit-learn

Install them using:
bash
pip install -r requirements.txt


---

## ⚙ How It Works
- Each implementation returns:
  - *weights (θ)*
  - *predicted values (y_pred)*
  - *Mean Squared Error (MSE)*  
- The main.py script:
  1. Calls each implementation  
  2. Collects the returned values  
  3. Passes the results to visualization.py for plotting & comparison  

---

## 🧮 Implemented Methods

### 1. Gradient Descent
An *iterative optimization algorithm* used to minimize the cost function.  
It updates the weights step by step in the opposite direction of the gradient.

*Update rule:*
\[
\theta := \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
\]

- Works well with large datasets  
- Requires choosing a learning rate (α) and number of iterations  
- May take time to converge  

---

### 2. Normal Equation
A *closed-form solution* for Linear Regression that directly computes the weights without iteration.

*Formula:*
\[
\theta = (X^T X)^{-1} X^T y
\]

- No need to choose learning rate or iterations  
- Works well for small to medium datasets  
- Becomes computationally expensive when number of features is very large (matrix inversion)  

---

### 3. Scikit-learn Implementation
Uses the **LinearRegression** class from Scikit-learn.  
Internally, it applies *Ordinary Least Squares (OLS)* which is mathematically equivalent to the Normal Equation, but implemented in an optimized way.

*Key differences from Normal Equation:*
- Handles numerical stability issues better  
- Efficient for large datasets due to optimized linear algebra libraries  
- Provides additional features (like regularization in Ridge/Lasso)  

python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)


---

## 📊 Results
- *Comparison Table* (example):

| Method            | MSE   | Notes                   |
|-------------------|-------|--------------------------|
| Gradient Descent  |  ...  | Iterative approach       |
| Normal Equation   |  ...  | Direct closed-form sol.  |
| Scikit-learn      |  ...  | Optimized library impl.  |

- *Visualizations*  
  Plots comparing predicted vs actual values for each method.  

---

## ⏱ Time Complexity Analysis

### 1. Gradient Descent
- Each iteration requires computing the gradient:
  \[
  O(m \cdot n)
  \]
  where:
  - \(m\) = number of training examples  
  - \(n\) = number of features  
- If we run for \(k\) iterations:
  \[
  O(k \cdot m \cdot n)
  \]
- *Scalable* for large \(n\) (features), but sensitive to choice of learning rate and number of iterations.

---

### 2. Normal Equation
- Requires computing:
  - \(X^T X\): \(O(m \cdot n^2)\)  
  - Inverting \((X^T X))\): \(O(n^3)\)  
- Total complexity:
  \[
  O(m \cdot n^2 + n^3)
  \]
- Efficient for *small to medium feature size (n)*, but becomes very slow if \(n\) is very large.

---

### 3. Scikit-learn Implementation
- Internally uses optimized *linear algebra solvers* (like LAPACK).  
- Complexity is mathematically similar to the *Normal Equation*:
  \[
  O(m \cdot n^2 + n^3)
  \]
- But due to optimized libraries, it is usually *faster and more stable* in practice.  
- Can handle larger datasets better than a manual Normal Equation implementation.

---

### 📊 Complexity Comparison Table

| Method            | Time Complexity            | Suitable for             |
|-------------------|----------------------------|--------------------------|
| Gradient Descent  | \(O(k \cdot m \cdot n)\)   | Large features (n) and datasets where iterative optimization is preferred |
| Normal Equation   | \(O(m \cdot n^2 + n^3)\)   | Small/medium datasets with fewer features |
| Scikit-learn      | \(O(m \cdot n^2 + n^3)\)   | Practical large datasets, optimized computation |

---

## 🎯 Key Takeaways
- *Gradient Descent*: scalable but slower to converge.  
- *Normal Equation*: simple but not efficient for large datasets.  
- *Scikit-learn*: best for practical use, highly optimized.  

---

## 📌 Future Work
- Try different datasets  
- Add Regularization (Ridge, Lasso)  
- Extend comparison to Polynomial Regression  

---
