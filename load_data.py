#!/usr/bin/env python3

"""
BI 410L Final Project. Implementation of SVM ML Model for E-Phys neural analysis.
Part One. Import and structure data.

Author: Julia De Geest
"""

import pandas as pd
import numpy as np
from collections import Counter
import bisect


def construct_neuron_spikes_dict() -> dict:
    """ Converts drifting_gratings_spikes (numpy npz object) to a more usable python dictionary.
    :param: drifting_gratings_spikes: numpy .npz object of drifting grating spikes

    :return: neuron_spikes_dict: dictionary of unique neurons and their corresponding firing times
    """
    # Load data
    drifting_gratings_spikes = np.load('drifting_gratings_spikes.npz')

    keys = drifting_gratings_spikes.files  # specify dict keys
    neuron_spikes_dict = {key: None for key in keys}  # instantiate dictionary

    # Add time values to dictionary
    for key in keys:
        neuron_spikes_dict[key] = drifting_gratings_spikes[key]

    return neuron_spikes_dict


def isolate_neurons(timestamp: tuple) -> list:
    """ Isolates the sequences that neurons fired for a specified time range:
    :param: timestamp: tuple that specifies the start and stop time of a time step
    :param: unique_neurons_list: list of unique neuron codes

    :return: neuron_sequence: list of neuron codes in the order that they fired for a time range.
    """
    # Construct dictionary of neurons and their firing times
    spikes_dict = construct_neuron_spikes_dict()

    # Define time variables
    start_time = timestamp[0]
    end_time = timestamp[1]

    for neuron in list(spikes_dict.keys()):
        spikes = spikes_dict[neuron]  # get all spike times for a neuron

        start_index = bisect.bisect_left(spikes, start_time)

        isolated_spikes = []  # holder for spikes that are within our specified time range
        for i in range(start_index, len(spikes)):
            if start_time <= spikes[i] <= end_time:  # if spike is in our time range...
                isolated_spikes.append(spikes[i])
            if spikes[i] > end_time:  # if spike_time is beyond our time range, break the loop
                break

        spikes_dict[neuron] = isolated_spikes  # reassign spike time to only those inside our time range

    # Isolate the neurons that had values, or in other words, fired. TRUE NAME.
    neurons_fired = [key for key, value in spikes_dict.items() if value]

    # Get corresponding most common areas for all neurons that fired
    neuron_areas = get_areas(neurons_fired)

    # Rename the neurons to be an integer 1 - n
    relabeled_spikes_dict = {}
    cnt = 0
    for value in spikes_dict.values():
        relabeled_spikes_dict[cnt] = value
        cnt += 1

    # Isolate the neurons that had values, or in other words, fired. RELABELED NAME.
    relabeled_neurons_fired = [key for key, value in relabeled_spikes_dict.items() if value]

    # Get top 20 neurons that fired
    sorted_neurons = sorted(relabeled_spikes_dict, key=lambda x: len(relabeled_spikes_dict[x]), reverse=True)
    top_20_neurons = sorted_neurons[:20]

    return top_20_neurons, neuron_areas


def get_areas(neurons_fired: list) -> list:
    """ Returns corresponding unique brain areas for neurons in list.
    :param: neurons_fired: list of neurons that fired in a certain time range

    :return: area_list: list of corresponding top 10 brain areas that fired
    """
    # Load areas data
    drifting_gratings_areas = np.load('drifting_gratings_areas.npz')

    # Get corresponding areas
    area_list = [drifting_gratings_areas[s].item() for s in neurons_fired]

    # Get 10 most common brain areas
    counts = Counter(area_list)
    best_areas = counts.most_common(n=10)  # 10 most common occurrences

    # Relabel the areas to correspond to integers
    all_areas = [drifting_gratings_areas[f].item() for f in drifting_gratings_areas.files]

    # To relabel our areas to integers, we can use it's index + 1 as a correlation.
    # We want to 0 for non-values, so lets start with a list of zeros
    relabeled_best_areas = np.zeros(10)
    for i in range(len(best_areas)):
        relabeled_best_areas[i] = all_areas.index(best_areas[i][0]) + 1

    return list(relabeled_best_areas)


def to_label(orientation: float) -> int:
    """ From angular orientation, returns corresponding label for classification

    :param orientation:
    :return: label: int
    """
    label = None  # initialize label

    if orientation == 0.0 or orientation == 180.0 or orientation == 360.0:
        label = 1

    elif orientation == 45.0 or orientation == 225.0:
        label = 2

    elif orientation == 90.0 or orientation == 270.0:
        label = 3

    elif orientation == 135.0 or orientation == 315.0:
        label = 4

    return label


def get_running_speed_info(timestamp: tuple) -> float:
    """ Get information (min, mean, median, max) for running speeds from a specific time point.

    :param: timestamp: tuple containing start and end time of a trial
    :return: minimum_speed, average_speed, median_speed, maximum_speed
    """
    # Load running speed data
    drifting_gratings_running = np.load('drifting_gratings_running.npz')

    # Define time variables
    start_time = timestamp[0]
    end_time = timestamp[1]

    # Take speed data from our wanted time range
    start_index = bisect.bisect_left(drifting_gratings_running['timestamps'], start_time)  # defines start index to not
    # loop through unnecessary values and run faster

    speeds = []
    for i in range(start_index, len(drifting_gratings_running['timestamps'])):
        current_time = drifting_gratings_running['timestamps'][i]
        if start_time <= current_time <= end_time:
            speeds.append(drifting_gratings_running['speed'][i])
        if current_time > end_time:  # if the current_time goes beyond our time range, break the loop
            break

    # Define wanted information about running speed
    minimum_speed = min(speeds)
    average_speed = np.mean(speeds)
    median_speed = np.median(speeds)
    maximum_speed = max(speeds)

    return minimum_speed, average_speed, median_speed, maximum_speed


def clean_data() -> None:
    """ Loads and compiles data from .npz files in local directory.

    :param: None. Requires the specific files below in local directory.
    :return: loaded and compiled data as pandas DataFrame
    """
    # Load our stimulus data
    drifting_gratings_stim = np.load('drifting_gratings_stim.npz')

    # Specify row data for each time point
    time_points = drifting_gratings_stim['start_time']

    # Instantiate an empty dictionary that will hold our row data
    data_df = pd.DataFrame(columns=list(np.arange(31)))

    # Create row data for each time point in the dataset
    for i in range(len(time_points)):
        row_data = []  # holder for individual row data

        time_tuple = (drifting_gratings_stim['start_time'][i], drifting_gratings_stim['stop_time'][i])  # time point

        temp_freq = drifting_gratings_stim['temporal_frequency'][i]  # temporal frequency
        if np.isnan(temp_freq):
            temp_freq = 0

        top_50_neurons, common_areas = isolate_neurons(time_tuple)  # neurons and most common areas
        min_, avg_, med_, max_ = get_running_speed_info(time_tuple)  # mouse running speed
        orientation = drifting_gratings_stim['orientation'][i]

        # Append the transformed data into a list to put in row of our dictionary
        # I chose to not include information on mouse speed or the temporal frequency because after test runs, it did
        # not help the overall accuracy.
        row_data = row_data + top_50_neurons
        row_data = row_data + common_areas
        row_data.append(to_label(orientation))

        # Handle case where process is manually stopped to not erase data.csv
        try:
            if to_label(orientation) is not None:  # do not export the row data if there is no orientation label
                data_df.loc[len(data_df)] = row_data
                print((i / 628) * 100)  # prints percentage of data structured

        except KeyboardInterrupt:
            print("Code process interrupted!")

        finally:
            if to_label(orientation) is not None:  # do not export the row data if there is no orientation label
                data_df.to_csv('data_new.csv')

    return None


clean_data()
