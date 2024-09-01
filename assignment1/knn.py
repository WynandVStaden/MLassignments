import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import sklearn
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
selector = SelectKBest(score_func=mutual_info_classif, k=17) 
X_new = selector.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE on the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for KNN using Randomized Search
knn_params = {'n_neighbors': range(3, 20), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
knn_random = RandomizedSearchCV(knn, knn_params, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_iter=20, random_state=42, n_jobs=-1)
knn_random.fit(X_train_scaled, y_train_resampled)
best_knn = knn_random.best_estimator_

# Evaluate the ensemble model
print("\nBest KNN Model Evaluation:")
predictions = best_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average='macro')
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score (Macro): {f1_macro:.2%}")
print(report)

classes = le_class.classes_

print("Best parameters found: ", knn_random.best_params_)

# The rest of your code remains the same

data = knn_random.cv_results_
results_df = pd.DataFrame(data)

# Sort the results by the mean test score in descending order
results_df = results_df.sort_values(by='mean_test_score', ascending=False)

# Extract relevant information for plotting
params = results_df['params']
mean_scores = results_df['mean_test_score']

# Separate parameters for better visualization
n_neighbors = [param['n_neighbors'] for param in params]
weights = [param['weights'] for param in params]
metric = [param['metric'] for param in params]

# Create a smaller figure suitable for LaTeX as PDF with x-axis as the number of neighbors
plt.figure(figsize=(6, 4))

# Create a scatter plot with n_neighbors on the x-axis
for i in range(len(params)):
    plt.scatter(n_neighbors[i], mean_scores.iloc[i],
                color='red' if metric[i] == 'euclidean' else 'blue', 
                marker='o' if weights[i] == 'uniform' else 'x')

# Add labels with smaller font sizes for LaTeX
plt.xlabel('Number of Neighbors (n_neighbors)', fontsize=8)
plt.ylabel('Mean Cross-Validated Accuracy', fontsize=8)
plt.title('Performance Across Different n_neighbors', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


# Adjust y-axis for better visibility
plt.ylim(min(mean_scores) - 0.001, max(mean_scores) + 0.001)

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Add a legend with smaller font sizes
red_patch1 = Line2D([0], [0], color='red', label='Euclidean and uniform weight', marker='o', linestyle='')
red_patch2 = Line2D([0], [0], color='red', label='Euclidean and distance weight', marker='x', linestyle='')   
blue_patch1 = Line2D([0], [0], color='blue', label='Manhattan and uniform weight', marker='o', linestyle='')   
blue_patch2 = Line2D([0], [0], color='blue', label='Manhattan and distance weight', marker='x', linestyle='')   


plt.legend(handles=[red_patch1, red_patch2, blue_patch1, blue_patch2])


# Ensure tight layout for LaTeX compatibility
plt.tight_layout()

# Save as a PDF for LaTeX
plt.savefig('parameter_performance_n_neighbors.pdf', format='pdf')

# Show the plot in the notebook
plt.show()

avg_fit_time = 0
avg_predict_time = 0
avg_fit_clock_speed = 0
avg_predict_clock_speed = 0
runs = 500

calculate_time = False
if calculate_time:
    for i in range(runs):
        # Measure CPU frequency before and after fitting
        fit_freq_start = psutil.cpu_freq().current
        fit_time_start = time.perf_counter()
        best_knn.fit(X_train, y_train)
        fit_time_end = time.perf_counter()
        fit_freq_end = psutil.cpu_freq().current

        # Measure CPU frequency before and after predicting
        predict_freq_start = psutil.cpu_freq().current
        predict_time_start = time.perf_counter()
        y_pred = best_knn.predict(X_test)
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

    avg_fit_time /= runs
    avg_predict_time /= runs
    avg_fit_clock_speed /= runs
    avg_predict_clock_speed /= runs


    print(f"(KNN) fit time: {avg_fit_time:.2f} ms, predict time: {avg_predict_time:.2f} ms")
    print(f"Average fit clock speed: {avg_fit_clock_speed:.2f} MHz, Average predict clock speed: {avg_predict_clock_speed:.2f} MHz")

# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')  # Display the confusion matrix using a blue colormap
plt.title("Confusion Matrix for Best KNN Model")
plt.show()