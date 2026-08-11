import os, sys, clr

# --- Setup Paths ---
user_profile = os.environ.get('USERPROFILE', r'C:\Users\guddu')
dwsim_path = os.path.join(user_profile, r"AppData\Local\DWSIM")
flowsheet_path = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\base_column.dwxmz"

sys.path.append(dwsim_path)
import System
from System.IO import Directory
Directory.SetCurrentDirectory(dwsim_path)

clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))
from DWSIM.Automation import Automation3

# --- Interrogate the File ---
print("Loading base file to check internal defaults...")
interf = Automation3()
sim = interf.LoadFlowsheet(flowsheet_path)

# WAKE UP THE MATH ENGINE
print("Calculating flowsheet to initialize thermodynamics...")
interf.CalculateFlowsheet(sim, None)

feed = sim.GetFlowsheetSimulationObject("feed").GetAsObject()

# Extract exactly what the file believes the baseline flows are
molar_flow = feed.GetMolarFlow()
mass_flow = feed.GetMassFlow()

print(f"\n--- BASEFILE FEED PROPERTIES ---")
print(f"Molar Flow: {molar_flow:.2f} mol/s")
print(f"Mass Flow:  {mass_flow:.4f} kg/s")