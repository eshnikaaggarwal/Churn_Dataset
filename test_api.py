import requests
import json

# Define the URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Create a dictionary with the necessary input data
data = {
    "international_plan": "yes",
    "voice_mail_plan": "no",
    "total_day_minutes": 150.0,
    "total_eve_minutes": 200.0,
    "total_night_minutes": 100.0,
    "total_day_charge": 25.5,
    "total_eve_charge": 34.0,
    "total_night_charge": 15.0,
    "total_intl_charge": 5.0,
    "state": "NY",
    "area_code": "415"
}

# Send a POST request to the Flask API
response = requests.post(url, json=data)

# Print the response from the server
print(response.json())
