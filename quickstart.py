# https://tsfresh.readthedocs.io/en/latest/text/quick_start.html

import matplotlib.pyplot as plt
from tsfresh.examples.robot_execution_failures import \
    download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features


###############
# Load data
###############

download_robot_execution_failures()

timeseries, y = load_robot_execution_failures()

print(timeseries.head())


###############
# Plot data
###############

timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True,
                                       figsize=(10, 10))
plt.show()


timeseries[timeseries['id'] == 20].plot(subplots=True, sharex=True,
                                        figsize=(10, 10))
plt.show()


##################
# Extract features
##################

extracted_features = extract_features(timeseries, column_id="id",
                                      column_sort="time")

print(extracted_features.shape)


##################
# Select features
##################

impute(extracted_features)
features_filtered = select_features(extracted_features, y)

print(features_filtered.shape)


####################################
# Extract and select in single step
####################################

features_filtered_direct = extract_relevant_features(timeseries, y,
                                                     column_id='id',
                                                     column_sort='time')

print(features_filtered_direct.shape)
