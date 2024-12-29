import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
# Ensure directories for saving graphics and CSV files
os.makedirs("D/IMERG/graphics", exist_ok=True)
os.makedirs("D/IMERG/CSV", exist_ok=True)
os.makedirs("D/IMERG", exist_ok=True)
# Load dataset
file_path = 'D/D.csv'
data = pd.read_csv(file_path)

# Step 1: Correlation Analysis
def plot_correlation(data, target_columns, save_path):
    correlation_matrix = data[target_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Target Columns')
    plt.savefig(save_path, dpi=300)
    plt.close()

#plot_correlation(data, ['GAUGE', 'CMORPH', 'GSMAP', 'IMERG'], 'D/correlation_matrix.png')

# Step 2: Neural Network Model
def build_and_train_nn(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(units=64, activation='relu', input_dim=X_train.shape[1]),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

X_nn = data[['Max_temp', 'Min_temp', 'RH', 'Max WS2m', 'Min WS2m', 'WD', 'WS10m', 'ST', 'SSWDI']]
y_nn = (data['IMERG'] > 0).astype(int)  # Binary target
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_nn, y_nn)

# Step 3: Split the data into training and testing sets
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler_nn = StandardScaler()
X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
X_test_nn_scaled = scaler_nn.transform(X_test_nn)

nn_model = build_and_train_nn(X_train_nn_scaled, X_test_nn_scaled, y_train_nn, y_test_nn)

# Neural Network Predictions
y_pred_nn_proba = nn_model.predict(X_test_nn_scaled).flatten()
y_pred_nn = (y_pred_nn_proba >= 0.5).astype(int)

# Step 3: Metrics Calculation
def calculate_metrics(y_true, y_pred, y_proba=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba) if y_proba is not None else "N/A"
    }

nn_metrics = calculate_metrics(y_test_nn, y_pred_nn, y_pred_nn_proba)

# Step 4: Other Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Single Layer Perceptron": MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=500, random_state=42)
}

results = {"Neural Network": nn_metrics}

for name, model in models.items():
    model.fit(X_train_nn_scaled, y_train_nn)
    y_pred = model.predict(X_test_nn_scaled)
    y_proba = model.predict_proba(X_test_nn_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    results[name] = calculate_metrics(y_test_nn, y_pred, y_proba)

# Step 5: Confusion Matrix Plot
def plot_conf_matrix(y_true, y_pred, model_name, save_path):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_conf_matrix(y_test_nn, y_pred_nn, "Neural Network", 'D/IMERG/graphics/conf_matrix_nn.png')

# Plot confusion matrices for other models
for name, model in models.items():
    y_pred = model.predict(X_test_nn_scaled)
    plot_conf_matrix(y_test_nn, y_pred, name, f'D/IMERG/graphics/conf_matrix_{name.replace(" ", "_").lower()}.png')

# Step 6: Visualization of Model Comparison
results_df = pd.DataFrame(results).T
results_df['accuracy'] *= 100  # Convert to percentage

plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=results_df.index, y=results_df['accuracy'], palette="viridis")

# Annotate accuracy scores
for i, acc in enumerate(results_df['accuracy']):
    bar_plot.text(i, acc + 1, f"{acc:.2f}%", ha='center', va='bottom')

plt.title("Model Comparison: Accuracy Scores")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("D/IMERG/graphics/model_comparison_accuracy.png", dpi=300)
plt.close()

# Save results to CSV
results_df.to_csv('D/IMERG/CSV/model_comparison_results.csv')

print("Model comparison results saved to 'D/IMERG/CSV/model_comparison_results.csv'.")
print("All graphics saved in 'D/IMERG/graphics/' directory.")
# Save Neural Network Predictions with Features
predictions_df = pd.DataFrame(X_test_nn, columns=X_nn.columns)  # Include feature names
predictions_df['True_Label'] = y_test_nn.values                 # Add True_Label
predictions_df['Predicted_Label'] = y_pred_nn.flatten()         # Add Predicted_Label
predictions_df['Predicted_Probability'] = y_pred_nn_proba.flatten()  # Add Predicted_Probability

# Save the DataFrame to CSV
predictions_df.to_csv('D/IMERG/CSV/nn_predictions_with_features.csv', index=False)

print("Neural Network predictions with features saved to 'D/IMERG/CSV/nn_predictions_with_features.csv'.")
# Step 5: Random Forest Regression: Feature Importance
X_rf = data.drop(columns=['GAUGE', 'CMORPH', 'GSMAP', 'IMERG'])
y_rf = data['IMERG']
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model_reg = RandomForestRegressor(random_state=42)
rf_model_reg.fit(X_train_rf, y_train_rf)

rf_feature_importances = pd.DataFrame({
    'Feature': X_rf.columns,
    'Importance': rf_model_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importances, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('D/IMERG/graphics/rf_feature_importance.png', dpi=300)
plt.close()