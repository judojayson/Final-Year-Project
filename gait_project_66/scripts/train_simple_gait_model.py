# train_simple_gait_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import time

# Load the combined processed gait dataset (66 features + label)
df = pd.read_csv("gait_project_66/data/processed/processed_gait_dataset.csv", header=None)
X = df.iloc[:, :-1].values  # 66 pose features
y = df.iloc[:, -1].values   # Labels: 0 = Normal, 1 = Cerebellar

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model (low complexity)
start_time = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_time = time.time() - start_time

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/simple_gait_model.pkl")

# Save accuracy and training time to a text file
with open("models/simple_gait_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Training Time: {train_time:.2f} seconds\n")

print("✅ Simple gait model saved as 'models/simple_gait_model.pkl'")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"⏱️ Training time: {train_time:.2f} seconds")
