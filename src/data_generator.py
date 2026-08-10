import os
import sys

# --- 1. Find DWSIM Automatically ---
user_profile = os.environ.get('USERPROFILE', r'C:\Users\guddu')

# List of common places DWSIM installs itself
possible_paths = [
    os.path.join(user_profile, r"AppData\Local\DWSIM"),
    os.path.join(user_profile, r"AppData\Local\DWSIM8"),
    r"C:\Program Files\DWSIM",
    r"C:\Program Files\DWSIM8"
]

dwsim_path = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "DWSIM.Automation.dll")):
        dwsim_path = path
        break

if dwsim_path is None:
    print("Could not find DWSIM installation automatically.")
    print("Please check where DWSIM.Automation.dll is located on your computer.")
    sys.exit()

print(f"Found DWSIM installation at: {dwsim_path}")

# --- 2. Force Load DWSIM Automation Libraries ---
import clr
sys.path.append(dwsim_path)

# CRUCIAL: We must change the directory so Windows can find the nested dependencies
import System
from System.IO import Directory
Directory.SetCurrentDirectory(dwsim_path)

try:
    # We pass the absolute path directly to force load it
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.GlobalSettings.dll"))
    print("Successfully connected to DWSIM DLLs!")
except Exception as e:
    print(f"Error loading DLLs: {e}")
    sys.exit()

from DWSIM.Automation import Automation3

# --- 3. Initialize DWSIM Headlessly ---
flowsheet_path = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\base_column.dwxmz"

print("Initializing DWSIM Engine...")
interf = Automation3()

print(f"Loading flowsheet from: {flowsheet_path}")
sim = interf.LoadFlowsheet(flowsheet_path)

if sim is None:
    print("Failed to load flowsheet. Please check if the file path is correct.")
    sys.exit()

# --- 4. Connect to Objects ---
feed = sim.GetFlowsheetSimulationObject("feed")
cond_duty = sim.GetFlowsheetSimulationObject("Q_C")

# Convert to usable Python/C# hybrid objects
feed_obj = feed.GetAsObject()
cond_obj = cond_duty.GetAsObject()

# --- 5. Run Test Simulation ---
print("\n--- Running Test Simulation ---")
new_temp = 310.0
print(f"Changing Feed Temperature to {new_temp} K...")

# The direct, crash-proof way to set temperature
feed_obj.SetTemperature(float(new_temp))

# Recalculate flowsheet headlessly
print("Calculating flowsheet in background...")
interf.CalculateFlowsheet(sim, None)

# Check results
if sim.Solved:
    # The direct way to pull the energy value in kW
    duty_val = cond_obj.EnergyFlow
    print(f"\nSUCCESS! Python successfully controlled DWSIM.")
    print(f"Calculated Condenser Duty (Q_C): {duty_val:.2f} kW")
else:
    print(f"Error solving flowsheet: {sim.ErrorMessage}")