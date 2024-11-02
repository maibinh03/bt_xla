import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dữ liệu mẫu
iris = load_iris()
X, y = iris.data, iris.target

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# CART - Gini Index
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# ID3 - Information Gain
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

# Hiển thị đánh giá
print("CART - Gini Index Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Confusion Matrix (CART):\n", confusion_matrix(y_test, y_pred_cart))
print("\nID3 - Information Gain Accuracy:", accuracy_score(y_test, y_pred_id3))
print("Confusion Matrix (ID3):\n", confusion_matrix(y_test, y_pred_id3))
