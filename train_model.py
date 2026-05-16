import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

print("Loading dataset...")
# It's actually a CSV despite the .xls extension
df = pd.read_csv('bank.xls')

# Define features and target
target_col = 'deposit'
X = df.drop(columns=[target_col])
y = df[target_col].apply(lambda x: 1 if x == 'yes' else 0)

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

print(f"Found {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features.")

# Create preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_cols)
    ])

# Create the full pipeline with the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Gradient Boosting model (this may take a few seconds)...")
model_pipeline.fit(X_train, y_train)

train_acc = model_pipeline.score(X_train, y_train)
test_acc = model_pipeline.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print("Saving model pipeline to gb_model.pkl...")
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Model saved successfully! Ready for Streamlit.")
