import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
import time
import psutil

# Load the dataset
file_path = 'assignment1/DryBeanDataSet.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Data Cleaning
df_clean = df.replace('?', np.nan)
imputer = SimpleImputer(strategy='most_frequent')
df_clean[['Colour', 'Class']] = imputer.fit_transform(df_clean[['Colour', 'Class']])

# Encode categorical columns
le_colour = LabelEncoder()
df_clean['Colour'] = le_colour.fit_transform(df_clean['Colour'])

le_class = LabelEncoder()
df_clean['Class'] = le_class.fit_transform(df_clean['Class'])

# Impute numeric columns
numeric_imputer = SimpleImputer(strategy='median')
df_clean.loc[:, df_clean.select_dtypes(include=[np.number]).columns] = numeric_imputer.fit_transform(df_clean.select_dtypes(include=[np.number]))


# Feature and target split
X = df_clean.drop('Class', axis=1)
y = df_clean['Class']


# Feature Selection using Information Gain (Shannon's Method)
# SelectKBest uses mutual information to select the top features
selector = SelectKBest(score_func=mutual_info_classif, k=17) 
X_new = selector.fit_transform(X, y)

# Get the scores of each feature
feature_scores = selector.scores_

# Display feature scores
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_scores})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Feature Importance based on Information Gain:")
print(feature_importance)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42, stratify=y)

# Use class weights to handle imbalance
dt_model = DecisionTreeClassifier(random_state=42, criterion='gini', class_weight='balanced')
dt_params = {'max_depth': range(3, 15), 'min_samples_split': range(2, 50), 'min_samples_leaf': range(1, 50), 'class_weight': ['balanced']}
#dt_params = {'max_depth': range(3, 15), 'class_weight': ['balanced']}
dt_random = RandomizedSearchCV(dt_model, dt_params, cv=StratifiedKFold(n_splits=3), scoring='accuracy', n_iter=50, random_state=42, n_jobs=-1)
dt_random.fit(X_train, y_train)
#Best parameters found:  {'min_samples_split': 16, 'min_samples_leaf': 7, 'max_depth': 11, 'class_weight': 'balanced'}
print("Best parameters found: ", dt_random.best_params_)

best_dt = dt_random.best_estimator_

# Evaluate the model
print("\nBest Decision Tree Model Evaluation:")
predictions = best_dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"F1 Score (Macro): {f1_score(y_test, predictions, average='macro'):.2%}")
print(classification_report(y_test, predictions))

classes = le_class.classes_

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Extract the results
results_df = pd.DataFrame(dt_random.cv_results_)

# Sort by mean test score for easier interpretation
results_df = results_df.sort_values(by='mean_test_score', ascending=False)

# Extract relevant information
params = results_df['params']
print(str(len(params)))
mean_scores = results_df['mean_test_score']

# Extract each hyperparameter for better visualization
max_depth = [param['max_depth'] for param in params]

# Create a plot with `max_depth` on the x-axis
plt.figure(figsize=(6, 4))

# Scatter plot, color and marker vary based on `min_samples_split` and `min_samples_leaf`
for i in range(len(params)):
    plt.scatter(max_depth[i], mean_scores.iloc[i],
                s=100, alpha=0.7, color='black')  # Increased marker size and added transparency

# Add labels with smaller font sizes for LaTeX
plt.xlabel('Max Depth (max_depth)', fontsize=8)
plt.ylabel('Mean Cross-Validated Accuracy', fontsize=8)
plt.title('Performance Across Different Max Depths', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Adjust y-axis for better visibility
plt.ylim(min(mean_scores) - 0.1, max(mean_scores) + 0.02)
# Ensure tight layout for LaTeX compatibility
plt.tight_layout()

# Save as a PDF for LaTeX
plt.savefig('decision_tree_performance_max_depth_improved.pdf', format='pdf')

# Show the plot
plt.show()

# Retrieve the best model
best_dt = dt_random.best_estimator_

avg_fit_time = 0
avg_predict_time = 0
avg_fit_clock_speed = 0
avg_predict_clock_speed = 0

calculate_time = False
if calculate_time:
    for i in range(500):
        # Measure CPU frequency before and after fitting
        fit_freq_start = psutil.cpu_freq().current
        fit_time_start = time.perf_counter()
        best_dt.fit(X_train, y_train)
        fit_time_end = time.perf_counter()
        fit_freq_end = psutil.cpu_freq().current

        # Measure CPU frequency before and after predicting
        predict_freq_start = psutil.cpu_freq().current
        predict_time_start = time.perf_counter()
        y_pred = best_dt.predict(X_test)
        predict_time_end = time.perf_counter()
        predict_freq_end = psutil.cpu_freq().current

        # Calculate times
        fit_time = (fit_time_end - fit_time_start) * 1000  # Convert seconds to milliseconds
        predict_time = (predict_time_end - predict_time_start) * 1000  # Convert seconds to milliseconds

        # Calculate average clock speed used during fitting and predicting
        fit_clock_speed = (fit_freq_start + fit_freq_end) / 2  # Average clock speed during fit
        predict_clock_speed = (predict_freq_start + predict_freq_end) / 2  # Average clock speed during predict

        avg_fit_time += fit_time
        avg_predict_time += predict_time
        avg_fit_clock_speed += fit_clock_speed
        avg_predict_clock_speed += predict_clock_speed

    avg_fit_time /= 500
    avg_predict_time /= 500
    avg_fit_clock_speed /= 500
    avg_predict_clock_speed /= 500

    print(f"(DT) fit time: {avg_fit_time:.2f} ms, predict time: {avg_predict_time:.2f} ms")
    print(f"Average fit clock speed: {avg_fit_clock_speed:.2f} MHz, Average predict clock speed: {avg_predict_clock_speed:.2f} MHz")

