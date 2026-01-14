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
SAMPLES = 10000 # no. of orbits to simulate
data = []

print(f"Generating {SAMPLES} orbits using C++ Engine...")

# using tqdm for progress bar
for i in tqdm(range(SAMPLES)):
    # static BH
    mass_bh = 1000.0

    # test particle (mix of stable and chaotic conditions)
    r = random.uniform(50, 150)
    # velocity with noise: v_circular = sqrt(GM/r)
    v_ideal = np.sqrt(1.0 * mass_bh / r)

    # randomize velocity factor
    # 0.5 = crash, 1.0 = circle, 1.5 = ellipse, 2.0 = escape
    v_factor = random.uniform(0.5, 2.5)

    v = v_ideal * v_factor

    # 2. PHYSICS-INFORMED LABELING
    # Instead of running a simulation, we calculate Total Energy.
    # E = KE + PE = 0.5*v^2 - GM/r
    
    kinetic_energy = 0.5 * (v**2)
    potential_energy = -1.0 * mass_bh / r  # G=1.0 in our engine
    total_energy = kinetic_energy + potential_energy
    
    # If Energy >= 0, it is Unbound (Unstable).
    # If Energy < 0, it is Bound (Stable).
    if total_energy >= 0:
        is_unstable = 1
    else:
        is_unstable = 0

    # record data
    data.append([r, v, mass_bh, is_unstable])

# -- save data --
df = pd.DataFrame(data, columns=["r", "v", "m_bh", "is_unstable"])
output_path = os.path.join(os.path.dirname(__file__), 'data/orbits.csv')
df.to_csv(output_path, index=False)

print(f"Done! Saved {len(df)} samples to {output_path}")
print(df["is_unstable"].value_counts()) # show balance (stable vs unstable)

