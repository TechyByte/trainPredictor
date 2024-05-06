from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime
import network_model
import logging
from meteostat import Hourly
from meteostat import Stations as WeatherStations  # avoiding ambiguity with [train] stations
import config
import requests
import os
# Load the saved model
model = load_model('trained_model.h5')

# Function to get current weather
# This is a placeholder, replace it with your actual function
def get_current_weather(tiploc):
    node = network_model.G.nodes(data=True)[tiploc]
    if "latlong" in node:
        # Extract latitude and longitude from the node
        latitude = node["latlong"][0]
        longitude = node["latlong"][1]

        # Parameters to include in the API call
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": config.openweather_API_key,
            "units": "metric",
            "exclude": "minutely,hourly,daily,alerts"
            # Excluding data not needed for this function to save data and time
        }

        response = requests.get(config.openweather_base_URL, params=params)
        print(response.request.url)
        if response.status_code == 200:
            data = response.json()
            current = data['current']

            temperature = current['temp']
            dew_point = current['dew_point']
            relative_humidity = current['humidity']
            precipitation = current.get('rain', {}).get('1h', 0) if 'rain' in current else 0
            snow = current.get('snow', {}).get('1h', 0) if 'snow' in current else 0
            wind_direction = current['wind_deg']
            wind_speed = current['wind_speed']
            peak_wind_gust = current.get('wind_gust', 0)
            pressure = current['pressure']
            # OpenWeather does not provide direct sunlight duration in current data, so this is left as None
            total_sun = None
            cloud_cover = current['clouds']

            return [
                temperature, dew_point, relative_humidity,
                precipitation, snow, wind_direction, wind_speed,
                peak_wind_gust, pressure, cloud_cover
            ]
        else:
            print("Failed to retrieve data")
            print(response.text)
            return [10,10,100,0,0,270,3,22,1018,4] # Placeholder data
    else:
        print("Latitude and longitude not found")
        return None



# Prepare the input data
incident_datetime = pd.to_datetime(datetime.now()).value  # Replace with your actual incident time
tiploc = "EXETRSD"  # Replace with your actual tiploc
train_service_code = '25474001'  # Replace with your actual train service code

# Get the current weather data
current_weather = get_current_weather(tiploc)

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder and transform the 'tiploc' and 'train_service_code' columns
tiploc = le.fit_transform([tiploc])[0]
train_service_code = le.fit_transform([train_service_code])[0]

# Now you can prepare the input data
input_data = np.array([[tiploc, incident_datetime, train_service_code] + current_weather])

# Make a prediction
prediction = model.predict(input_data)

print(f'The predicted delay is: {prediction[0][0]}')