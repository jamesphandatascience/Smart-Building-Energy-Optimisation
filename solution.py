'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: James Phan   zID: z5360539

'''
# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle
import json
from DiscreteFactors import Factor
from HiddenMarkovModel import HiddenMarkovModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Helper Functions

# Binarize sensor dataframes
def binarize_sensor_df(df):
    binary_df = df.copy()

    for col in df.columns:
        if col.startswith('motion_sensor'):
            # 1 if motion detected, else 0
             binary_df[col] = df[col].apply(lambda x: 1 if str(x) == 'motion' else 0)

        elif col.startswith('door_sensor'):
            # 1 if door sensor counted someone (positive int), else 0
            binary_df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

        elif col.startswith('camera'):
            # 1 if camera sees anyone (>0), else 0
            binary_df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

        elif col.startswith('wifi_sensor'):
            # Already binary (0 or 1), but ensure it's integer
            binary_df[col] = df[col].apply(lambda x: 1 if str(x) == 'device detected' else 0)

        else:
            # Leave other columns (e.g., room_r, timestamp) unchanged
            continue

    return binary_df

# Binarize sensor_data

def binarize_sensor_data(sensor_data):
    binary_data = {}

    for key, value in sensor_data.items():
        if key.startswith('motion_sensor'):
            # 1 if motion detected, else 0
            binary_data[key] = 1 if str(value) == 'motion' else 0

        elif key.startswith('door_sensor'):
            # 1 if positive count, else 0
            binary_data[key] = 1 if isinstance(value, (int, float)) and value > 0 else 0

        elif key.startswith('camera'):
            # 1 if count > 0, else 0
            binary_data[key] = 1 if isinstance(value, (int, float)) and value > 0 else 0

        elif key.startswith('wifi_sensor'):
            # 1 if device detected, else 0
            binary_data[key] = 1 if str(value) == 'device detected' else 0

        else:
            # Keep as is
            binary_data[key] = value

    return binary_data

###################################

# Load datasets
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")
data = pd.concat([data1, data2], axis=0, ignore_index=True)

# Extract rooms data and convert to binary values for occupied (1), not occupied (0)
room_cols = [f"r{i}" for i in range(1, 35)]

room_df1 = data1[room_cols]
room_df2 = data2[room_cols]
room_df = data[room_cols]

# Calculate expected number of people when occupied
expected_people_if_occupied = {}

for col in room_df.columns:
    occupied_counts = room_df.loc[room_df[col] > 0, col]
    expected_people_if_occupied[col] = occupied_counts.mean()

room_df1 = room_df1.map(lambda x: 1 if x > 0 else 0)
room_df2 = room_df2.map(lambda x: 1 if x > 0 else 0)
room_df = room_df.map(lambda x: 1 if x > 0 else 0)

# Binarize sensor df
binarized_df1 = binarize_sensor_df(data1)
binarized_df2 = binarize_sensor_df(data2)
binarized_df = binarize_sensor_df(data)

# Infer transition matrix for each room from data 
transition_counts = {'r' + str(room): np.zeros((2, 2)) for room in range(1,35)}

for room in room_cols:
    room1_states = room_df1[room].values
    room2_states = room_df2[room].values

    for i in range(len(room1_states) - 1):
        curr1 = room1_states[i]
        nexx1 = room1_states[i + 1]
        
        curr2 = room2_states[i]
        nexx2 = room2_states[i + 1]
        
        transition_counts[room][curr1][nexx1] += 1
        transition_counts[room][curr2][nexx2] += 1
    
transition_factors = {}

for room, matrix in transition_counts.items(): 
    row_sums = matrix.sum(axis=1, keepdims=True)
    prob_matrix = matrix / row_sums
    
    # force unoccupied row to [0.5, 0.5]
    # prob_matrix[0, :] = [0.3, 0.7]

    outcomeSpace = {
        'state': ('unoccupied', 'occupied'),
        'next_state': ('unoccupied', 'occupied')
    }
    factor = Factor(('state', 'next_state'), outcomeSpace, table=prob_matrix)
    transition_factors[room] = factor



# Infer starting state belief for each room from data 
start_state_factors = {}
for room, counts in transition_counts.items():
    counts = np.array(counts)
    num_meetings = counts[0, 1]
    p_occ = num_meetings / 4798

    # State outcome space
    outcomeSpace = {'state': ('unoccupied', 'occupied')}

    # Table is 1D vector of probabilities
    prob_table = np.array([1-p_occ, p_occ])
    # prob_table = np.array([0.1, 0.9])

    # Create the Factor
    factor = Factor(('state',), outcomeSpace, table=prob_table)
    start_state_factors[room] = factor

#Custom emissions based on floor plan
room_sensors = {
    'r1': ['motion_sensor1', 'door_sensor4'],
    'r2': ['camera2', 'door_sensor4'],
    'r3': ['door_sensor4', 'wifi_sensor2'],
    'r4': ['door_sensor9'],
    'r5': ['door_sensor9'],
    'r6': ['wifi_sensor4', 'door_sensor9'],
    'r7': ['door_sensor7'],
    'r8': ['door_sensor7'],
    'r9': ['door_sensor7'],
    'r10': ['door_sensor7'],
    'r11': ['door_sensor7'],
    'r12': ['door_sensor7'],
    'r13': ['door_sensor7'],
    'r14': ['door_sensor9', 'motion_sensor5'],
    'r15': ['door_sensor1'],
    'r16': ['door_sensor2'],
    'r17': ['door_sensor5'],
    'r18': ['door_sensor6'],
    'r19': ['door_sensor3', 'motion_sensor2'],
    'r20': ['door_sensor3', 'camera3'],
    'r21': ['wifi_sensor3', 'door_sensor8'],
    'r22': ['wifi_sensor5', 'door_sensor9'],
    'r23': ['camera1', 'door_sensor3'],
    'r24': ['wifi_sensor6', 'door_sensor10'],
    'r25': ['wifi_sensor1', 'door_sensor8'],
    'r26': ['door_sensor8'],
    'r27': ['camera4', 'door_sensor8'],
    'r28': ['motion_sensor6', 'door_sensor10'],
    'r29': ['motion_sensor3', 'door_sensor8'],
    'r30': ['door_sensor8'],
    'r31': ['door_sensor8'],
    'r32': ['motion_sensor4', 'door_sensor8'],
    'r33': ['door_sensor8'],
    'r34': ['door_sensor8']
}

# room_sensors = {
#     'r1':  ['motion_sensor1', 'door_sensor4', 'wifi_sensor1'],
#     'r2':  ['camera2', 'door_sensor4', 'motion_sensor2'],
#     'r3':  ['door_sensor4', 'wifi_sensor2', 'motion_sensor3'],
#     'r4':  ['door_sensor9', 'motion_sensor4', 'wifi_sensor3'],
#     'r5':  ['door_sensor9', 'motion_sensor5', 'wifi_sensor4'],
#     'r6':  ['wifi_sensor4', 'door_sensor9', 'camera1'],
#     'r7':  ['door_sensor7', 'motion_sensor6', 'wifi_sensor5'],
#     'r8':  ['door_sensor7', 'motion_sensor2', 'wifi_sensor6'],
#     'r9':  ['door_sensor7', 'motion_sensor3', 'wifi_sensor1'],
#     'r10': ['door_sensor7', 'motion_sensor4', 'wifi_sensor2'],
#     'r11': ['door_sensor7', 'motion_sensor5', 'wifi_sensor3'],
#     'r12': ['door_sensor7', 'motion_sensor6', 'wifi_sensor4'],
#     'r13': ['door_sensor7', 'motion_sensor1', 'wifi_sensor5'],
#     'r14': ['door_sensor9', 'motion_sensor5', 'wifi_sensor6'],
#     'r15': ['door_sensor1', 'motion_sensor2', 'wifi_sensor1'],
#     'r16': ['door_sensor2', 'motion_sensor3', 'wifi_sensor2'],
#     'r17': ['door_sensor5', 'motion_sensor4', 'wifi_sensor3'],
#     'r18': ['door_sensor6', 'motion_sensor5', 'wifi_sensor4'],
#     'r19': ['door_sensor3', 'motion_sensor2', 'wifi_sensor5'],
#     'r20': ['door_sensor3', 'camera3', 'motion_sensor6'],
#     'r21': ['wifi_sensor3', 'door_sensor8', 'motion_sensor1'],
#     'r22': ['wifi_sensor5', 'door_sensor9', 'motion_sensor2'],
#     'r23': ['camera1', 'door_sensor3', 'motion_sensor3'],
#     'r24': ['wifi_sensor6', 'door_sensor10', 'motion_sensor4'],
#     'r25': ['wifi_sensor1', 'door_sensor8', 'motion_sensor5'],
#     'r26': ['door_sensor8', 'wifi_sensor2', 'motion_sensor6'],
#     'r27': ['camera4', 'door_sensor8', 'motion_sensor1'],
#     'r28': ['motion_sensor6', 'door_sensor10', 'wifi_sensor3'],
#     'r29': ['motion_sensor3', 'door_sensor8', 'wifi_sensor4'],
#     'r30': ['door_sensor8', 'motion_sensor4', 'wifi_sensor5'],
#     'r31': ['door_sensor8', 'motion_sensor5', 'wifi_sensor6'],
#     'r32': ['motion_sensor4', 'door_sensor8', 'wifi_sensor1'],
#     'r33': ['door_sensor8', 'motion_sensor6', 'wifi_sensor2'],
#     'r34': ['door_sensor8', 'motion_sensor1', 'wifi_sensor3'],
# }


# emission_factors = {}

# for room, sensors in room_sensors.items():
#     # Variables: state + sensors
#     domain_vars = ['state'] + sensors
#     num_vars = len(domain_vars)

#     # Count table: N-dimensional binary table
#     count_table = np.zeros([2] * num_vars, dtype=int)

#     state_vals = room_df[room].values
#     sensor_vals = [binarized_df[s].values for s in sensors]

#     # Count occurrences
#     for i in range(len(state_vals)):
#         idx = tuple([state_vals[i]] + [sensor_vals[j][i] for j in range(len(sensors))])
#         count_table[idx] += 1

#     # Normalize P(sensors | state)
#     prob_table = np.zeros_like(count_table, dtype=float)
#     for state_val in [0, 1]:
#         state_slice = count_table[state_val]
#         total = state_slice.sum()
#         if total > 0:
#             alpha = 1e-6
#             smoothed_slice = state_slice + alpha
#             prob_table[state_val] = smoothed_slice / smoothed_slice.sum()
#         else:
#             prob_table[state_val] = 1.0 / state_slice.size  # uniform if no data

#     # Build outcome space
#     outcome_space = {'state': ('unoccupied', 'occupied')}
#     for var in sensors:
#         outcome_space[var] = (0, 1)

#     # Create Factor
#     factor = Factor(tuple(domain_vars), outcome_space, table=prob_table)
#     emission_factors[room] = factor

#     print(factor)


emission_factors = {}

# Smoothing parameter
alpha = 1.0
boost_factor = 3.0  # optional boosting for occupancy indicators

for room, sensors in room_sensors.items():
    state_vals = room_df[room].values
    sensor_vals = {s: binarized_df[s].values for s in sensors}

    # Build outcome space
    outcome_space = {'state': ('unoccupied', 'occupied')}
    for s in sensors:
        outcome_space[s] = (0, 1)

    num_sensors = len(sensors)
    num_vars = 1 + num_sensors  # state + sensors

    # Count table for all combinations: shape (2, 2, 2, ...) depending on num_sensors
    count_table = np.zeros([2] * num_vars, dtype=float)

    # Count occurrences
    for i in range(len(state_vals)):
        idx = tuple([state_vals[i]] + [sensor_vals[s][i] for s in sensors])
        count = 1.0

        # Optional boosting: if occupied and any strong sensor=1, multiply
        if state_vals[i] == 1 and any(sensor_vals[s][i] == 1 for s in sensors):
            count *= boost_factor

        count_table[idx] += count

    # Normalize per state to get P(sensors | state)
    prob_table = np.zeros_like(count_table)
    for state_val in [0, 1]:
        state_slice = count_table[state_val]
        # Add smoothing
        state_slice += alpha
        prob_table[state_val] = state_slice / state_slice.sum()

    # Create Factor
    factor = Factor(
        ('state',) + tuple(sensors),  # domain (positional)
        outcome_space,                # outcomeSpace (positional)
        table=prob_table              # table (keyword ok)
   )

    emission_factors[room] = factor


# Build HMMS for each room
hmms = {}
for room in room_cols:
    variable_remap = {
        'next_state': 'state'   # maps the new state variable to the current
    }
    hmms[room] = HiddenMarkovModel(
        start_state=start_state_factors[room],
        transition=transition_factors[room],
        emission=emission_factors[room],
        variable_remap=variable_remap
    )

def get_action(sensor_data):
    global hmms
    global expected_people_if_occupied
    
    binary_data = binarize_sensor_data(sensor_data)
    actions_dict = {}
    
    for room_name, room_hmm in hmms.items():
        # Identify this room's emission variables
        room_vars = [var for var in room_hmm.emission.domain if var != 'state']
        
        # Extract evidence for this room
        room_evidence = {var: binary_data[var] for var in room_vars if var in binary_data}
        
        # Update HMM belief
        belief = room_hmm.forward(**room_evidence)
    
        
        # Get P(occupied) from the belief distribution
        p_occupied = belief['occupied']
        p_unoccupied = belief['unoccupied']

        alpha = 1e-12

        total = belief['occupied'] + belief['unoccupied']
        if total == 0:
            p_occupied, p_unoccupied = 0.5, 0.5  # fallback uniform
        else:
            p_occupied = (belief['occupied'] + alpha) / (total + 2*alpha)
            p_unoccupied = (belief['unoccupied'] + alpha) / (total + 2*alpha)
        
        # print(f"p_ouccupied is {p_occupied}, p_unouccupied is {p_unoccupied}")

        # Expected number of people if occupied
        e_people = expected_people_if_occupied[room_name]
        
        # Compute cost of lights off/on
        cost_off = p_occupied * 4 * 20
        cost_on = p_unoccupied

        # Decision rule
        if cost_off > cost_on:
            actions_dict[f"lights{room_name[1:]}"] = 'on'
        else:
            actions_dict[f'lights{room_name[1:]}'] = 'off'
        
    
    # Robots override light controls (assuming 100% accuracy from specs)


    if binary_data['robot1'] is not None:   
        match = re.match(r"\('([^']+)',\s*(\d+)\)", binary_data['robot1'])
        if match:
            room = match.group(1)
            people = int(match.group(2))
    
        if room.startswith('r'):
        
            if people > 0:
                actions_dict[f'lights{room[1:]}'] = 'on'
            # if people == 0:
            #     actions_dict[f'lights{room[1:]}'] = 'off'

    
    if binary_data['robot2'] is not None:   
        match = re.match(r"\('([^']+)',\s*(\d+)\)", binary_data['robot2'])
        if match:
            room = match.group(1)
            people = int(match.group(2))
        
        if room.startswith('r'):
            
            if people > 0:
                actions_dict[f'lights{room[1:]}'] = 'on'
            # if people == 0:
            #     actions_dict[f'lights{room[1:]}'] = 'off'
    
    return actions_dict
