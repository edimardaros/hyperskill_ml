# 130Exersise - Diabetes detection - Linear Regression with scikit-learn
# https://hyperskill.org/learn/step/15042


from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

