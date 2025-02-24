import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("CharlesBookClub.csv")

# Data Preprocessing
# Dropping unnecessary columns (adjust based on dataset structure)
df.drop(columns=["Seq#", "ID#"], inplace=True)

# Checking for missing values
print("Missing values:\n", df.isnull().sum())

# Splitting features and target variable
X = df.drop(columns=["Florence"])  # Features
y = df["Florence"]  # Target variable

# Splitting into training and testing datasets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_resampled)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Model Evaluation:")
print(classification_report(y_test, y_pred_knn))

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train_resampled)
y_pred_log = log_reg.predict(X_test_scaled)
print("Logistic Regression Model Evaluation:")
print(classification_report(y_test, y_pred_log))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_resampled, y_train_resampled)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Model Evaluation:")
print(classification_report(y_test, y_pred_dt))

# Random Forest (Best Model)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)
print("Random Forest Model Evaluation:")
print(classification_report(y_test, y_pred_rf))

# Visualization - Feature Importance (Random Forest)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='royalblue')
plt.title('Top 10 Important Features - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')

# Save the plot before showing
plt.savefig("feature_importance.png", bbox_inches='tight', dpi=300)
plt.show()

# Visualization - Feature Importance (Random Forest)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.show()

# Visualization - Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

