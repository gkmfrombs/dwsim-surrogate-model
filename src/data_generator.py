import os
import sys
import csv
import time
from scipy.stats import qmc

# --- 1. DWSIM Setup & Connection ---
user_profile = os.environ.get('USERPROFILE', r'C:\Users\guddu')
dwsim_path = os.path.join(user_profile, r"AppData\Local\DWSIM")
flowsheet_path = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\base_column.dwxmz"
output_csv = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\results.csv"

import clr
sys.path.append(dwsim_path)
import System
from System.IO import Directory
Directory.SetCurrentDirectory(dwsim_path)

try:
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))
    from DWSIM.Automation import Automation3
except Exception as e:
    print(f"Error loading DLLs: {e}")
    sys.exit()

interf = Automation3()

# --- 2. Latin Hypercube Sampling (LHS) ---
n_samples = 5000
print(f"Generating {n_samples} LHS samples...")

# 7 input variables
sampler = qmc.LatinHypercube(d=7)
sample_points = sampler.random(n=n_samples)

# Bounds: [Temp, Press, Benzene_Frac, Stages, Feed_Stage, Reflux, Bottoms_Rate]
l_bounds = [290.0, 100000.0, 0.3, 15, 5, 1.2, 40.0]
u_bounds = [350.0, 150000.0, 0.7, 30, 14, 5.0, 60.0]
scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)

# --- 3. Prepare CSV ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
headers = [
    "Run_ID", "Feed_Temp_K", "Feed_Press_Pa", "Benzene_Frac", "Stages", 
    "Feed_Stage", "Reflux_Ratio", "Bottoms_Rate", 
    "x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW", "Converged"
]

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# --- 4. The Automation Loop ---
print("\nStarting automated simulations...")
start_time = time.time()

for i, point in enumerate(scaled_samples):
    sim = interf.LoadFlowsheet(flowsheet_path)
    
    feed = sim.GetFlowsheetSimulationObject("feed").GetAsObject()
    col = sim.GetFlowsheetSimulationObject("DCOL-1").GetAsObject()
    distillate = sim.GetFlowsheetSimulationObject("Top_Product").GetAsObject()
    bottoms = sim.GetFlowsheetSimulationObject("Bottom_Product").GetAsObject()
    cond_duty = sim.GetFlowsheetSimulationObject("Q_C").GetAsObject()
    reb_duty = sim.GetFlowsheetSimulationObject("Q_R").GetAsObject()

    t_feed, p_feed, bz_frac = point[0], point[1], point[2]
    stages, f_stage = int(point[3]), int(point[4])
    reflux, b_rate = point[5], point[6]
    
    if f_stage >= stages:
        f_stage = stages - 2 

    try:
        # 1. Change feed conditions
        feed.SetTemperature(float(t_feed))
        feed.SetPressure(float(p_feed))
        
        from System import Array, Double
        comp_array = Array[Double]([float(bz_frac), float(1.0 - bz_frac)])
        feed.SetOverallComposition(comp_array)
        
        # 2. Change column structural rules (Stages) safely
        # Use the explicit method to force DWSIM to rebuild the internal floor matrix
        col.SetNumberOfStages(stages)
        
        # Now safely attach the pipe
        col.SetStreamFeedStage(feed.Name, f_stage)

        # 3. Set mathematical specifications
        col.Specs["C"].SpecValue = float(reflux)
        col.Specs["R"].SpecValue = float(b_rate)
        
        # 4. Calculate
        interf.CalculateFlowsheet(sim, None)
        
        if sim.Solved:
            x_D = distillate.GetPhaseComposition(0)[0] 
            x_B = bottoms.GetPhaseComposition(0)[0]
            Q_C = cond_duty.EnergyFlow
            Q_R = reb_duty.EnergyFlow
            converged = True
        else:
            x_D, x_B, Q_C, Q_R = 0, 0, 0, 0
            converged = False

    except Exception as e:
        print(f"CRASH ON RUN {i+1}: {e}")
        x_D, x_B, Q_C, Q_R = 0, 0, 0, 0
        converged = False

    # Checkpoint to CSV
    row = [
        i+1, t_feed, p_feed, bz_frac, stages, f_stage, reflux, b_rate, 
        x_D, x_B, Q_C, Q_R, converged
    ]
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
        
    # GC cleanup
    sim = None
    System.GC.Collect()
        
    # Print every 50 run for the test
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Completed {i + 1}/{n_samples} runs. Time elapsed: {elapsed:.1f}s")

print(f"\nData generation complete! Results saved to {output_csv}")