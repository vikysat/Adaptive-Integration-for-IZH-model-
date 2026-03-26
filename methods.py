from models import izhNeuron
import numpy as np
import time

def reference(neuron: izhNeuron, I:np.ndarray, steps: int):
    dt = 0.001
    v = np.zeros(steps)
    u = np.zeros(steps)
    ddv = np.zeros(steps)
    ddu = np.zeros(steps)
    
    start_time = time.time()
    for i in range(0, steps, int(dt*1000)):

        if neuron.v >= 30:
            neuron.v = neuron.c
            neuron.u += neuron.d

        dv = (0.04 * neuron.v ** 2 + 5 * neuron.v + 140 - neuron.u + I[i]) * dt
        du = (neuron.a * (neuron.b * neuron.v - neuron.u)) * dt
        neuron.v += dv
        neuron.u += du

        try:
            ddv[i] = dv/dt
        except IndexError:
            pass
        try:
            ddu[i] = du/dt
        except IndexError:
            pass

        v[i] = neuron.v
        u[i] = neuron.u

    end_time = time.time()
    print(f"Reference implementation took {end_time - start_time:.4f} seconds")
        
    return v, u, end_time - start_time

def default(neuron: izhNeuron, I:np.ndarray, steps: int):
    dt = 0.025
    v = np.zeros(steps)
    u = np.zeros(steps)
    
    start_time = time.time()
    for i in range(0, steps, int(dt*1000)):

        if neuron.v >= 30:
            neuron.v = neuron.c
            neuron.u += neuron.d

        dv = (0.04 * neuron.v ** 2 + 5 * neuron.v + 140 - neuron.u + I[i]) * dt
        du = (neuron.a * (neuron.b * neuron.v - neuron.u)) * dt
        neuron.v += dv
        neuron.u += du

        neuron.v = min(neuron.v, 30)

        for j in range(int(dt*1000)):
            v[i+j] = neuron.v
            u[i+j] = neuron.u
        
    end_time = time.time()
    print(f"Default implementation took {end_time - start_time:.4f} seconds")
        
    return v, u, end_time - start_time

# Make modified verson here:

def adaptivesig(neuron: 'izhNeuron', I: np.ndarray, steps: int,k,v0):

    v_out = np.zeros(steps)
    dt_out = np.zeros(steps)
    u_out = np.zeros(steps)

    max_jump = 100
    min_jump = 25

    # sigmoid parameters (tune these)
     # steepness      # center of transition (where dt starts shrinking)

    curr_idx = 0
    start_time = time.time()

    while curr_idx < steps:
        v_old, u_old = neuron.v, neuron.u

        # -------- SIGMOID-BASED ADAPTIVE RULE --------
        sig = 1 / (1 + np.exp(-k * (v_old - v0)))
        jump = int(min_jump + (max_jump - min_jump) * (1 - sig))
        jump = max(min_jump, min(max_jump, jump))
        # ---------------------------------------------

        if curr_idx + jump >= steps:
            jump = (steps - 1) - curr_idx
            if jump <= 0:
                break

        dt = jump * 0.001
        current_I = I[curr_idx]

        dv = (0.04 * v_old**2 + 5 * v_old + 140 - u_old + current_I) * dt
        du = (neuron.a * (neuron.b * v_old - u_old)) * dt
        
        v_new = v_old + dv
        u_new = u_old + du

        if v_new >= 30:
            alpha = (30 - v_old) / (v_new - v_old)
            spike_offset = int(round(alpha * jump))
            spike_idx = curr_idx + spike_offset

            v_out[spike_idx] = 30.0
            u_out[spike_idx] = u_old + alpha * du 

            neuron.v = neuron.c
            neuron.u += neuron.d

            v_out[curr_idx: spike_idx] = v_old
            u_out[curr_idx: spike_idx] = u_old
            dt_out[curr_idx: spike_idx] = dt

            curr_idx = spike_idx + 1
        else:
            v_out[curr_idx: curr_idx + jump] = v_new
            u_out[curr_idx: curr_idx + jump] = u_new
            dt_out[curr_idx: curr_idx + jump] = dt

            neuron.v = v_new
            neuron.u = u_new
            curr_idx += jump

    end_time = time.time()
    print(f"Adaptive dt implementation took {end_time - start_time:.4f} seconds")

    return v_out, u_out, end_time - start_time

def adaptive_dt(neuron: 'izhNeuron', I: np.ndarray, steps: int,k,c):

    v_out = np.zeros(steps)
    dt_out = np.zeros(steps)
    u_out = np.zeros(steps)

    max_jump = 100
    min_jump = 25   
    
    curr_idx = 0
    start_time = time.time()
    
    while curr_idx < steps:
        v_old, u_old = neuron.v, neuron.u

        if v_old > -50:
            jump = int(max(min_jump, max_jump / (1 + k * abs(v_old + c))))
        else:
            jump = max_jump
            
        if curr_idx + jump >= steps:
            jump = (steps - 1) - curr_idx
            if jump <= 0: break

        dt = jump * 0.001
        current_I = I[curr_idx]

        dv = (0.04 * v_old**2 + 5 * v_old + 140 - u_old + current_I) * dt
        du = (neuron.a * (neuron.b * v_old - u_old)) * dt
        
        v_new = v_old + dv
        u_new = u_old + du

        if v_new >= 30:
            alpha = (30 - v_old) / (v_new - v_old)
            spike_offset = int(round(alpha * jump))
            spike_idx = curr_idx + spike_offset
            
            v_out[spike_idx] = 30.0
            u_out[spike_idx] = u_old + alpha * du 
            
            neuron.v = neuron.c
            neuron.u += neuron.d
            
            v_out[curr_idx : spike_idx] = v_old
            u_out[curr_idx : spike_idx] = u_old
            dt_out[curr_idx: spike_idx] = dt

            curr_idx = spike_idx + 1
        else:
            v_out[curr_idx : curr_idx + jump] = v_new
            u_out[curr_idx : curr_idx + jump] = u_new
            dt_out[curr_idx: curr_idx+jump] = dt
            
            neuron.v = v_new
            neuron.u = u_new
            curr_idx += jump
        


    end_time = time.time()
    print(f"Adaptive dt implementation took {end_time - start_time:.4f} seconds")
    return v_out, u_out, end_time - start_time



def interpolated(neuron: 'izhNeuron', I: np.ndarray, steps: int):
    dt = 0.025
    v_out = np.zeros(steps)
    u_out = np.zeros(steps)
    v_peak = 30.0
    
    start_time = time.time()
    chunk = int(dt * 1000)
    
    for i in range(0, steps, chunk):
        v_old = neuron.v
        u_old = neuron.u
        current_I = I[i]

        dv = (0.04 * v_old**2 + 5 * v_old + 140 - u_old + current_I)
        du = (neuron.a * (neuron.b * v_old - u_old))
        
        v_new = v_old + dv * dt
        u_new = u_old + du * dt

        actual_chunk = min(chunk, steps - i)

        if v_new >= v_peak:
            alpha = (v_peak - v_old) / (v_new - v_old)
            u_at_spike = u_old + alpha * (du * dt)

            v_out[i] = v_peak 
            u_out[i] = u_at_spike
            
            dt_rem = (1 - alpha) * dt
            v_reset = neuron.c
            u_reset = u_at_spike + neuron.d

            dv_rem = (0.04 * v_reset**2 + 5 * v_reset + 140 - u_reset + current_I)
            du_rem = (neuron.a * (neuron.b * v_reset - u_reset))
            
            neuron.v = v_reset + dv_rem * dt_rem
            neuron.u = u_reset + du_rem * dt_rem
            
            if actual_chunk > 1:
                v_out[i+1 : i + actual_chunk] = neuron.v
                u_out[i+1 : i + actual_chunk] = neuron.u
        else:
            neuron.v = v_new
            neuron.u = u_new
            v_out[i : i + actual_chunk] = neuron.v
            u_out[i : i + actual_chunk] = neuron.u
        
    end_time = time.time()
    print(f"Interpolated implementation took {end_time - start_time:.4f} seconds")
    return v_out, u_out, end_time - start_time







def adaptive_dvdt_exp(neuron: 'izhNeuron', I: np.ndarray, steps: int, k):

    v_out = np.zeros(steps)
    dt_out = np.zeros(steps)
    u_out = np.zeros(steps)

    max_jump = 100
    min_jump = 25   
    
    curr_idx = 0
    start_time = time.time()
    
    while curr_idx < steps:
        v_old, u_old = neuron.v, neuron.u
        current_I = I[curr_idx]

        # -------- compute dv/dt ----------
        dv_dt = 0.04*v_old**2 + 5*v_old + 140 - u_old + current_I

        # -------- exponential rule --------
        jump = int(
            min_jump +
            (max_jump - min_jump) * np.exp(-k * abs(dv_dt))
        )
        # ----------------------------------

        if curr_idx + jump >= steps:
            jump = (steps - 1) - curr_idx
            if jump <= 0:
                break

        dt = jump * 0.001

        dv = dv_dt * dt
        du = (neuron.a * (neuron.b * v_old - u_old)) * dt
        
        v_new = v_old + dv
        u_new = u_old + du

        if v_new >= 30:
            alpha = (30 - v_old) / (v_new - v_old)
            spike_offset = int(round(alpha * jump))
            spike_idx = curr_idx + spike_offset
            
            v_out[spike_idx] = 30.0
            u_out[spike_idx] = u_old + alpha * du 
            
            neuron.v = neuron.c
            neuron.u += neuron.d
            
            v_out[curr_idx:spike_idx] = v_old
            u_out[curr_idx:spike_idx] = u_old
            dt_out[curr_idx:spike_idx] = dt

            curr_idx = spike_idx + 1
        else:
            v_out[curr_idx:curr_idx + jump] = v_new
            u_out[curr_idx:curr_idx + jump] = u_new
            dt_out[curr_idx:curr_idx + jump] = dt
            
            neuron.v = v_new
            neuron.u = u_new
            curr_idx += jump

    end_time = time.time()
    print(f"Adaptive dv/dt exponential took {end_time - start_time:.4f} seconds")

    return v_out, u_out, end_time - start_time, dt_out


