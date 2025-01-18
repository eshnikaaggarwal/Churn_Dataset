from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('churn_model.pkl')

# Load processed training data to get feature columns
processed_train = pd.read_csv('processed_train.csv')
feature_columns = processed_train.drop('churn', axis=1).columns.tolist()

def preprocess_input(data):
    # Convert JSON data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Handle missing values if any
    input_df.fillna(method='ffill', inplace=True)
    
    # Convert 'yes'/'no' to 1/0 for categorical fields
    input_df['international_plan'] = input_df['international_plan'].map({'yes': 1, 'no': 0})
    input_df['voice_mail_plan'] = input_df['voice_mail_plan'].map({'yes': 1, 'no': 0})
    
    # One-Hot Encoding for 'state' and 'area_code'
    input_encoded = pd.get_dummies(input_df, columns=['state', 'area_code'], drop_first=True)
    
    # Ensure all training features are present
    missing_cols = set(feature_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0  # Add missing columns with default value 0
    
    # Ensure the order of columns matches the training data
    input_encoded = input_encoded[feature_columns]
    
    return input_encoded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    try:
        # Preprocess the input data
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return the result
        return jsonify({'predicted_churn': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
