import logging

import requests

import config
import network_model


def placeholder_weather():
    return [10, 10, 100, 0, 0, 270, 3, 22, 1018, 40, 4]  # Placeholder data

def get_current_weather(tiploc):
    node = network_model.G.nodes(data=True)[tiploc]
    if "latlong" in node:
        # Parameters to include in the API call
        params = {
            "lat": node["latlong"][0],
            "lon": node["latlong"][1],
            "appid": config.openweather_API_key,
            "units": "metric",
            "exclude": "minutely,hourly,daily,alerts"
            # Excluding data not needed for this function to save data and time
        }

        response = requests.get(config.openweather_base_URL, params=params)

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
            logging.debug(f"Failed to retrieve weather data for {tiploc}. Response: " + response.text)
            return placeholder_weather()
    else:
        print(f"Latitude and longitude not found for {tiploc}")
        return None

#TODO: def get_weather_forecast(tiploc, date):
#TODO:     include weather caching
