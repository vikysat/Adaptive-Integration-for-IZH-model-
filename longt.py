import matplotlib.pyplot as plt
import numpy as np
from models import izhNeuron, FS, RS, LTS, CH, TC
from methods import reference, default, adaptive_dt,adaptivesig,interpolated,adaptive_dvdt_exp
import csv

# ----------------- Load input -----------------
with open('Iext_long.csv', 'r') as f:
    reader = csv.reader(f)
    I = np.array([float(row[0]) for row in reader])
steps = len(I)

# ----------------- Run reference -----------------
neuron_ref = RS()
v_ref, u_ref, time_def = reference(neuron_ref, I, steps)
v_def, u_def, time_def = interpolated(neuron_ref, I, steps)
v_dt,u_dt,time_dt = adaptive_dt(neuron_ref, I, steps,0.19,50)

# ----------------- Run adaptive solver (example k, v0) -----------------
neuron_adp = RS()
v_adp, u_adp, time_ada,t = adaptive_dvdt_exp(neuron_adp, I, steps,1.05)

v_sig, u_sig, time_sig = adaptivesig(neuron_ref, I, steps,0.13,-60)



# ----------------- Compute MAE -----------------

"""
voltage_error = np.abs(v_def - v_adp)
mae = np.mean(voltage_error)
print("Mean voltage error (mV):", mae)

# ----------------- Time vector -----------------
time = np.arange(0, steps * 0.001, 0.001)

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(time[:len(v_def)], v_def, label='Reference Voltage')
plt.plot(time[:len(v_adp)], v_adp, label='Adaptive Voltage', color='orange')
plt.title('Voltage over Time')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.tight_layout()
plt.show()
"""
# ----------------- RMSD function -----------------
def compute_rmsd(v1, v2):
    """Compute root mean square deviation between two voltage traces."""
    return np.sqrt(np.mean((v1 - v2) ** 2))


print("exp",compute_rmsd(v_ref,v_adp))
print("interpolated",compute_rmsd(v_def,v_ref))
print("regular",compute_rmsd(v_ref,v_dt))
print("sigmoid",compute_rmsd(v_ref,v_sig))

# ----------------- Optimize k -----------------

"""
def optimize_v0(I, steps, v0_values, k=0.13):
    results = []

    # Run default/reference solver once
    neuron_ref = RS()
    v_ref, _, _ = reference(neuron_ref, I, steps)

    best_rmsd = float("inf")
    best_v0 = None

    for v0 in v0_values:
        neuron_adp = RS()
        v_adp, _, _ = adaptivesig(neuron_adp, I, steps, k, v0)

        rmsd = compute_rmsd(v_ref, v_adp)
        results.append((v0, rmsd))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_v0 = v0

        print(f"v0 = {v0:.2f}, RMSD = {rmsd:.4f}")

    print("\nBest v0 found:", best_v0)
    print("Lowest RMSD:", best_rmsd)

    return best_v0, best_rmsd, results


# ----------------- Run optimization -----------------
v0_values = np.linspace(-60, -30, 20)   # sweep possible center points
best_v0, best_rmsd, results = optimize_v0(I, steps, v0_values, k=0.13)


# Extract RMSD for plotting
ks, rmsds = zip(*results)



plt.subplot(2,1,2)
plt.plot(ks, rmsds, '-o', color='blue')
plt.xlabel("k (sigmoid steepness)")
plt.ylabel("RMSD (mV)")
plt.title("RMSD vs k for Adaptive Solver")
plt.grid(True)
plt.tight_layout()
plt.show()

"""