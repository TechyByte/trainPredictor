import config

import logging
import pickle
import argparse
import networkx
import pandas as pd
import numpy as np

# from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense
from keras.layers import Dropout, BatchNormalization, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logging.info(f"Loading historical model: {config.historical_model_filename}")
with open(config.historical_model_filename, "rb") as file:
    raw_model: networkx.Graph = pickle.load(file)

# Extract features from the raw_model
features = []
labels = []

count = 0
total = len(raw_model.nodes)
success = 0



# Create the parser
parser = argparse.ArgumentParser(
    description='Perform learning on all nodes or a specific node. Leave blank to learn on all nodes or specify TIPLOC.')

# Add the arguments
parser.add_argument('--node', type=str,
                    help='The node to perform learning on. If not provided, learning will be performed on all nodes.')

# Parse the arguments
args = parser.parse_args()

# Extract the node argument (if provided)
node_to_learn = args.node  # Set to a specific node to only learn from that node

logging.info(
    "Model will be saved as trained_model_" + (node_to_learn if node_to_learn is not None else "whole") + '.h5')

logging.info(f"Extracting training data from raw model...")

for node, data in raw_model.nodes(data=True):
    count += 1
    if count % int(total / 100) == 0:
        logging.info(f"Processing nodes: {100 * count / total:0.4f}% ({count} of {total})")

    # If a specific node is provided and the current node is not the specified node, skip this iteration
    if node_to_learn is not None and node != node_to_learn:
        continue

    if "incidents" in data and "weather_history" in data:
        incidents = data["incidents"]
        weather_history = data["weather_history"]
        weather_history.index = pd.to_datetime(weather_history.index).astype('int64')
        # weather_history.index = weather_history.index.to_pydatetime()

        # For each incident, find the corresponding weather data
        for _, incident in incidents.iterrows():
            incident_datetime = pd.to_datetime(incident['INCIDENT_START_DATETIME'])

            incident_datetime_int64 = incident_datetime.value

            if not weather_history.empty:
                # Find the weather data entry that is closest in time to the incident
                i = np.argmin(np.abs(weather_history.index - incident_datetime_int64))

                closest_weather = weather_history.iloc[i].fillna(0).astype('int64')

                # Combine the incident data and the corresponding weather data
                combined_features = np.concatenate([[str(node), incident_datetime_int64],
                                                    incident[['TRAIN_SERVICE_CODE']]
                                                   .values, closest_weather.values.flatten()])

                # Add the combined features to the list
                features.append(combined_features)

                # Determine the label for the incident
                label = max(incident['PFPI_MINUTES'], incident['NON_PFPI_MINUTES'])  # , incident['EVENT_TYPE'],
                # incident['INCIDENT_REASON']

                labels.append(label)
                success += 1
            else:
                logging.debug(f"No weather history data available for node {node}")

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

logging.info("Features and labels extracted")

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder and transform the 'EXETRSD' column
features[:, 0] = le.fit_transform(features[:, 0])

# Check if features is empty
if features.size == 0:
    print("Error: No features were extracted. Please check the conditions for appending to the 'features' list.")
    exit()

# Normalize the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

logging.info("Splitting data into training and validation sets...")
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
logging.info("Complete")
# Combine y_train and y_val
# y_combined = np.concatenate((y_train, y_val), axis=0)

# Fit the LabelEncoder on the combined data and transform the labels for each column
# le.fit(np.array([column for column in y_combined.T]))
# y_train = np.array([le.transform(column) for column in y_train.T]).T
# y_val = np.array([le.transform(column) for column in y_val.T]).T
logging.info("Preparing to train...")
# Define the model
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Reshape((128, 1)))
model.add(Conv1D(filters=128, kernel_size=9,
                 activation='sigmoid'))  # , input_shape=(X_train.shape[1], 1)))  # Increased number of filters
model.add(Dropout(0.1))

# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(LSTM(units=100, return_sequences=True,
               activation='sigmoid'))
model.add(Dropout(0.1))
# model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# Reshape the data to fit the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

logging.info("Training model...")
# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=256)
logging.info("Training complete")
# Display the training loss and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the training set
train_loss = model.evaluate(X_train, y_train, verbose=0)

# Evaluate the model on the validation set
val_loss = model.evaluate(X_val, y_val, verbose=0)

print(f'Training Loss: {train_loss}')
print(f'Validation Loss: {val_loss}')

# Make predictions
predictions = model.predict(X_val)

print(predictions)

# Save the model
model.save('trained_model_' + (node_to_learn if node_to_learn is not None else "whole") + '.h5')
print("Model saved as trained_model_" + (node_to_learn if node_to_learn is not None else "whole") + '.h5')

from sklearn.metrics import confusion_matrix

# Round the predictions to the nearest integer
rounded_predictions = np.round(predictions)

# Generate the confusion matrix
matrix = confusion_matrix(y_val, rounded_predictions)

print(matrix)
