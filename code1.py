import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ---------------------------
# 1. Load the Dataset
# ---------------------------

# Update the path to your local file
df = pd.read_csv("train.csv")

# Show the dataset
print(df.head())

print('Dataset info:')
df_info = df.info()  # Removed inline print for better readability
print(df_info)

# Dataset summary statistics
print(df.describe())

# Handle missing values by forward filling
df.fillna(method='ffill', inplace=True)

# ---------------------------
# 2. Handle Categorical Variables
# ---------------------------

# Columns to encode
categorical_cols = ['state', 'area_code', 'international_plan', 'voice_mail_plan', 'churn']

# Initialize LabelEncoder
label_encoder = preprocessing.LabelEncoder()

# Label Encoding for 'international_plan', 'voice_mail_plan', 'churn'
df['international_plan'] = label_encoder.fit_transform(df['international_plan'])
df['voice_mail_plan'] = label_encoder.fit_transform(df['voice_mail_plan'])
df['churn'] = label_encoder.fit_transform(df['churn'])

print('for international_plan :', df['international_plan'].unique())
print('for voice_mail_plan :', df['voice_mail_plan'].unique())
print('for churn :', df['churn'].unique())

# One-Hot Encoding for 'state' and 'area_code'
df_encoded = pd.get_dummies(df, columns=['state', 'area_code'], drop_first=True)

print(df_encoded.head())
print(df_encoded.dtypes)

# ---------------------------
# 3. Calculate Correlation Matrix
# ---------------------------

correlation_matrix = df_encoded.corr(numeric_only=True)  # Ensures only numeric columns are included
print(correlation_matrix['churn'].sort_values(ascending=False))

# ---------------------------
# 4. Visualization
# ---------------------------

# a. Correlation Matrix Heatmap
plt.figure(figsize=(40, 20))
sns.heatmap(
    correlation_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    linewidths=0.5,
    annot=True,  # Adding correlation values to the heatmap
    fmt=".2f"    # Correlation values to 2 decimal places
)
plt.title('Correlation Matrix of Telecom Churn Dataset')
plt.show()

# b. Boxplots for Main Features
main_columns = ['international_plan', 'number_customer_service_calls', 'total_day_minutes', 'total_day_charge']

for i in main_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_encoded[i])
    plt.title(f'Boxplot of {i}')
    plt.show()

# ---------------------------
# 5. Outlier Detection using Z-score
# ---------------------------

features = ['international_plan', 'number_customer_service_calls', 'total_day_minutes', 'total_day_charge']

for j in features:
    mean = df_encoded[j].mean()
    median = df_encoded[j].median()
    std_dev = df_encoded[j].std()

    # Z-score method to find outliers
    z_scores = np.abs((df_encoded[j] - mean) / std_dev)
    outliers = (z_scores > 3).sum()  # Common threshold for outliers

    print(f"{j}:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Median: {median:.4f}")
    print(f"  Standard Deviation: {std_dev:.4f}")
    print(f"  Number of Outliers (Z-score method): {outliers}\n")

# ---------------------------
# 6. Outlier Removal using IQR Method
# ---------------------------

def remove_outliers(df_encoded, i):
    Q1 = df_encoded[i].quantile(0.25)
    Q3 = df_encoded[i].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df_encoded[(df_encoded[i] >= lower_bound) & (df_encoded[i] <= upper_bound)]

# Define the columns from which to remove outliers
columns_to_change = ['international_plan', 'number_customer_service_calls', 'total_day_minutes', 'total_day_charge']

# Remove outliers for the specified columns
for col in columns_to_change:
    df_encoded = remove_outliers(df_encoded, col)

# Print the updated DataFrame info to verify changes
print(df_encoded.info())

# ---------------------------
# 7. Remove Duplicate Rows
# ---------------------------

duplicates = df_encoded.duplicated()
print(df_encoded[duplicates])
df_encoded = df_encoded.drop_duplicates()

# ---------------------------
# 8. Feature Creation
# ---------------------------

df_encoded['total_minutes'] = (
    df_encoded['total_day_minutes'] + 
    df_encoded['total_eve_minutes'] + 
    df_encoded['total_night_minutes']
)
df_encoded['total_charges'] = (
    df_encoded['total_day_charge'] + 
    df_encoded['total_eve_charge'] + 
    df_encoded['total_night_charge'] + 
    df_encoded['total_intl_charge']
)
print(df_encoded[['total_minutes', 'total_charges']].head())

# Dropping redundant columns
columns_drop = [
    'total_day_minutes', 'total_eve_minutes', 'total_night_minutes',
    'total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge'
]
df_encoded = df_encoded.loc[:, ~df_encoded.columns.isin(columns_drop)]
print(df_encoded.head())

# ---------------------------
# 9. Churn Distribution Analysis
# ---------------------------

# Check for the number of yes and no in the churn column
churn_number = df_encoded['churn'].value_counts()
print('Distribution of churn is:', churn_number)

# Plot the churn_number variable for better understanding
churn_number.plot(kind='bar', color=['blue', 'green'])
plt.title("Churn Class Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# Churn Percentage
churn_percent = df_encoded['churn'].value_counts() / len(df_encoded) * 100
print(churn_percent)

# ---------------------------
# 10. Train-Test Split and Model Training
# ---------------------------

X = df_encoded.drop(columns=['churn'])  # Features
y = df_encoded['churn']               # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# ---------------------------
# 11. Predicting on Test Dataset
# ---------------------------

test_data = pd.read_csv('test.csv')  # Ensure 'test.csv' is in the same directory
test_data.fillna(method='ffill', inplace=True)

test_data_encoded = pd.get_dummies(test_data, columns=['state', 'area_code'], drop_first=True)

# Feature Engineering
test_data_encoded['total_minutes'] = (
    test_data_encoded['total_day_minutes'] + 
    test_data_encoded['total_eve_minutes'] + 
    test_data_encoded['total_night_minutes']
)
test_data_encoded['total_charges'] = (
    test_data_encoded['total_day_charge'] + 
    test_data_encoded['total_eve_charge'] + 
    test_data_encoded['total_night_charge'] + 
    test_data_encoded['total_intl_charge']
)

# Convert 'yes' and 'no' values to 1 and 0 in the test data
test_data_encoded['international_plan'] = test_data_encoded['international_plan'].map({'yes': 1, 'no': 0})
test_data_encoded['voice_mail_plan'] = test_data_encoded['voice_mail_plan'].map({'yes': 1, 'no': 0})

# Ensure X_test has the same columns and is in the same order as X_train
missing_cols = set(X_train.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0  # Add missing columns with default value 0

test_data_encoded = test_data_encoded[X_train.columns]

# Make predictions
predictions = model.predict(test_data_encoded)

# Add predictions to the original test_data
test_data['predicted_churn'] = predictions
test_data.to_csv('predicted_churn_results.csv', index=False)

# Display the distribution of predicted churn
predicted_data = pd.read_csv('predicted_churn_results.csv')
print(predicted_data['predicted_churn'].value_counts())

# ---------------------------
# 12. Model Deployment
# ---------------------------

# Save the trained model to a file
joblib.dump(model, 'churn_model.pkl')

# Save processed training data
df_encoded.to_csv('processed_train.csv', index=False)
print("Processed training data saved to 'processed_train.csv'")
