"""
Replay a recorded day of sensor data through the lighting controller and compare its cost
with two simple baselines: lights always on, and lights always off.

Cost model (from the assignment brief), per room and per 15-second step:
  - light on:  1 cent of electricity
  - light off: 4 cents of lost productivity for every person in the room

Usage:
    python evaluate.py            # evaluates data1.csv and data2.csv
    python evaluate.py data2.csv  # evaluates one day
"""
import datetime
import importlib
import sys

import numpy as np
import pandas as pd

ROOMS = [f"r{i}" for i in range(1, 35)]
ELECTRICITY_COST = 1
PRODUCTIVITY_COST = 4


def sensor_columns(df):
    prefixes = ("motion_sensor", "door_sensor", "camera", "wifi_sensor", "robot")
    return [c for c in df.columns if c.startswith(prefixes)]


def evaluate_day(path, seed=0):
    np.random.seed(seed)
    import solution
    importlib.reload(solution)  # reset every room's HMM belief to its start state

    day = pd.read_csv(path)
    sensors = sensor_columns(day)
    lights = {f"lights{i}": "off" for i in range(1, 35)}
    clock = datetime.time(8, 0)

    cost = lit_steps = occupied_steps = occupied_lit = 0
    for _, row in day.iterrows():
        reading = {c: row[c] for c in sensors}
        reading[np.random.choice(sensors)] = None  # the simulator breaks one random sensor each step
        clock = (datetime.datetime.combine(datetime.date.today(), clock) + datetime.timedelta(seconds=15)).time()
        reading["time"] = clock

        lights.update(solution.get_action(reading))

        for light, state in lights.items():
            people = row["r" + light[6:]]
            if state == "on":
                cost += ELECTRICITY_COST
                lit_steps += 1
            else:
                cost += PRODUCTIVITY_COST * people
            if people > 0:
                occupied_steps += 1
                occupied_lit += state == "on"

    room_steps = len(day) * len(ROOMS)
    always_on = room_steps * ELECTRICITY_COST
    always_off = int(PRODUCTIVITY_COST * day[ROOMS].to_numpy().sum())
    return {
        "day": path,
        "HMM controller": int(cost),
        "always on": always_on,
        "always off": always_off,
        "saving vs always on": f"{1 - cost / always_on:.1%}",
        "energy used vs always on": f"{lit_steps / room_steps:.1%}",
        "occupied rooms lit": f"{occupied_lit / occupied_steps:.1%}",
    }


if __name__ == "__main__":
    days = sys.argv[1:] or ["data1.csv", "data2.csv"]
    results = pd.DataFrame([evaluate_day(d) for d in days]).set_index("day")
    print(results.T.to_string())
