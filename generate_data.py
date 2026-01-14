import sys
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm # progress-bar

# load universe c++ engine 
sys.path.append(os.path.join(os.path.dirname(__file__), '../HybridEngine/build'))

try:
    import gravity_core
    print("SUCCESS: Loaded C++ Hybrid Engine!")
except ImportError:
    print("ERROR: Could not load 'Hybrid Engine'. Check your path!")
    sys.exit(1)

# CONFIGURATION
SAMPLES = 5000 # no. of orbits to simulate
STEPS = 5000 # duration of simulation of each orbit
DT = 0.01 # time step
ESCAPE_RADIUS = 200.0 # threshold radius of orbit to flag as "Ejected"

data = []

print(f"Generating {SAMPLES} orbits using C++ Engine...")

# using tqdm for progress bar
for i in tqdm(range(SAMPLES)):
    # initialize universe (1 BH + 1 test particle)
    sim = gravity_core.Universe(2)

    # static BH
    mass_bh = 1000.0

    # test particle (mix of stable and chaotic conditions)
    r = random.uniform(50, 150)
    angle = random.uniform(0, 6.28)

    px_0 = r * np.cos(angle)
    py_0 = r * np.sin(angle)

    # velocity with noise: v_circular = sqrt(GM/r)
    v_ideal = np.sqrt(1.0 * mass_bh / r)

    # randomize velocity factor
    # 0.5 = crash, 1.0 = circle, 1.5 = ellipse, 2.0 = escape
    v_factor = random.uniform(0.5, 2.5)

    vx_0 = -v_ideal * np.sin(angle) * v_factor
    vy_0 = -v_ideal * np.cos(angle) * v_factor

    # set state by passing lists
    sim.set_state(
        [0.0, px_0], # x (bh, particle)
        [0.0, py_0], # y
        [0.0, vx_0], # vel. in x
        [0.0, vy_0], # vel. in y
        [mass_bh, 1.0] # mass of bh, particle
    )
    
    # run simulation
    is_unstable = 0

    # we run in chunks to check for escape
    for _ in range(10): # check 10 times during the simulation
        for _ in range(STEPS // 10):
            sim.step()

        # check position
        curr_x = sim.get_x()[1]
        curr_y = sim.get_y()[1]
        dist = np.sqrt(curr_x**2 + curr_y**2)

        if dist > ESCAPE_RADIUS:
            is_unstable = 1
            break # stop early if particles flies off

    # record data
    data.append([px_0, py_0, vx_0, vy_0, mass_bh, is_unstable])

# -- save data --
df = pd.DataFrame(data, columns=["px", "py", "vx", "vy", "m_bh", "label"])
output_path = os.path.join(os.path.dirname(__file__), 'data/orbits.csv')
df.to_csv(output_path, index=False)

print(f"Done! Saved {len(df)} samples to {output_path}")
print(df["label"].value_counts()) # show balance (stable vs unstable)

