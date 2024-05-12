
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import date, datetime
from sklearn.preprocessing import LabelEncoder

import weather
from weather import get_current_weather
import schedule

# Load the saved model
model = load_model('trained_model_EXETRSD.h5')

services = schedule.get_scheduled_services(date.today(), "EXETRSD")

input_data = pd.DataFrame(columns=["tiploc", "time", "train_service_code", "weather"])
count = 0

for service in services:
    count += 1
    movements = service.get_movements()
    for movement in movements:
        train_service_code = service.tsc
        try:
            current_weather = get_current_weather(movement.tiploc)
            new_row = pd.DataFrame({"tiploc": [movement.tiploc], "time": [date.today() + movement.time],
                                    "train_service_code": [service.tsc], "weather": [current_weather]})
            input_data = pd.concat([input_data, new_row], ignore_index=True)
        except KeyError:
            new_row = pd.DataFrame({"tiploc": [movement.tiploc], "time": [date.today() + movement.time],
                                    "train_service_code": [service.tsc], "weather": [weather.placeholder_weather()]})
            input_data = pd.concat([input_data, new_row], ignore_index=True)
    if count > 5:
        break

# Initialize the LabelEncoder
le = LabelEncoder()


# Fit the LabelEncoder and transform the 'tiploc' and 'train_service_code' columns
input_data["tiploc"] = le.fit_transform(input_data["tiploc"])
#input_data["train_service_code"] = le.fit_transform(input_data["train_service_code"])


# Convert the Unix timestamp to int64
input_data['time'] = pd.to_datetime(input_data['time']).dt.tz_localize(None)
for i in range(len(input_data)):
    input_data.loc[i, 'time'] = input_data.loc[i, 'time'].timestamp()

# Convert 'time' column to numeric format
input_data['time'] = pd.to_datetime(input_data['time']).dt.tz_localize(None)
input_data['time'] = input_data['time'].apply(lambda x: x.timestamp()).astype('float32')

# Convert 'weather' column to float32
#input_data['weather'] = input_data['weather'].apply(lambda x: [float(i) for i in x])

# Now, you can convert these specific columns to float32 and create the input_data_array
input_data_array = np.column_stack([input_data['tiploc'].values, input_data['time'].values, input_data['train_service_code'].values, input_data['weather'].values.tolist()]).astype('float32')

# Make a prediction
prediction = model.predict(input_data_array)



print(prediction)
assert len(input_data["time"]) == len(prediction)
print(f'The predicted delay is {prediction[0][0][0]} minutes')
