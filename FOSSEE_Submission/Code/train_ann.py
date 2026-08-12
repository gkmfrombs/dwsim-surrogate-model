import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ---------------------------------------------------------
# 1. THE NEURAL NETWORK ARCHITECTURE ("The Factory")
# ---------------------------------------------------------
class SurrogateANN(nn.Module):
    def __init__(self, input_size=7, output_size=4):
        super(SurrogateANN, self).__init__()
        # Layer 1: Takes the 7 inputs and expands them to 64 hidden neurons
        self.layer1 = nn.Linear(input_size, 64)
        # Activation function: Adds non-linearity (helps the network learn curves, not just straight lines)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 64 hidden neurons -> 64 hidden neurons
        self.layer2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        
        # Output Layer: 64 hidden neurons -> 4 final predictions
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        # This defines how the data flows through the factory
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

# ---------------------------------------------------------
# 2. THE PHYSICS-INFORMED LOSS FUNCTION ("The Strict Manager")
# ---------------------------------------------------------
def physics_loss_function(predictions, targets, inputs, scaler_X, scaler_y, standard_mse_loss):
    """
    Calculates standard error, but ADDS a penalty if the network breaks the law of conservation of mass.
    """
    # 1. Calculate standard math error (Mean Squared Error)
    mse = standard_mse_loss(predictions, targets)
    
    # 2. Un-scale the data back to real-world units so we can do physics math
    # We detach from the graph temporarily to use numpy, but for the penalty gradient to flow, 
    # we must do the math using PyTorch tensors.
    
    # Convert scalers to PyTorch tensors
    mean_X = torch.tensor(scaler_X.mean_, dtype=torch.float32)
    scale_X = torch.tensor(scaler_X.scale_, dtype=torch.float32)
    mean_y = torch.tensor(scaler_y.mean_, dtype=torch.float32)
    scale_y = torch.tensor(scaler_y.scale_, dtype=torch.float32)
    
    # Real Inputs = (Scaled Inputs * scale) + mean
    real_inputs = (inputs * scale_X) + mean_X
    # Real Predictions = (Scaled Predictions * scale) + mean
    real_preds = (predictions * scale_y) + mean_y
    
    # 3. Physics Math: Law of Conservation of Mass (F*z = D*x_D + B*x_B)
    # Feed Flow (F) was fixed at 100 mol/s in our simulations
    F = 100.0
    
    # From Inputs:
    z = real_inputs[:, 2]         # Benzene_Frac is column index 2
    B = real_inputs[:, 6]         # Bottoms_Rate is column index 6
    D = F - B                     # What doesn't go to the bottom goes to the top
    
    # From Predictions:
    x_D = real_preds[:, 0]        # predicted x_D_Benzene is column index 0
    x_B = real_preds[:, 1]        # predicted x_B_Benzene is column index 1
    
    benzene_in = F * z
    benzene_out = (D * x_D) + (B * x_B)
    
    # The Physics Penalty is how much Benzene mysteriously "vanished" or "appeared"
    physics_penalty = torch.mean(torch.abs(benzene_in - benzene_out))
    
    # Total Loss = Standard Math Error + (Weight * Physics Penalty)
    # We heavily weight the physics penalty so the network takes it seriously!
    total_loss = mse + (0.1 * physics_penalty)
    
    return total_loss

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
def train_ann():
    print("--- 1. Loading Preprocessed Data ---")
    data_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\processed"
    models_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\models"
    
    # Load Scalers
    scaler_X = joblib.load(os.path.join(models_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(models_dir, "scaler_y.pkl"))

    # Load Data
    X_train_df = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    X_test_df = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test_df = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    
    # Convert Data to PyTorch Tensors
    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32)
    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_df.values, dtype=torch.float32)
    
    # Initialize the Network
    model = SurrogateANN()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # The tool that updates the weights
    standard_mse_loss = nn.MSELoss()
    
    print("--- 2. Training the Physics-Informed Neural Network (PINN) ---")
    epochs = 500
    for epoch in range(epochs):
        # Forward Pass: Make a guess
        predictions = model(X_train)
        
        # Calculate Loss (Math Error + Physics Penalty)
        loss = physics_loss_function(predictions, y_train, X_train, scaler_X, scaler_y, standard_mse_loss)
        
        # Backward Pass: The Manager yells at the workers, forcing them to adjust
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Calculate how much to adjust each worker
        optimizer.step()      # Actually make the adjustment
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f}")
            
    print("\n--- 3. Evaluating on the Hidden Test Set ---")
    # Turn off training mode (locks the workers in place)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    
    overall_mse = mean_squared_error(y_test.numpy(), y_pred)
    overall_r2 = r2_score(y_test.numpy(), y_pred)
    
    print(f"Overall Mean Squared Error (MSE): {overall_mse:.4f}")
    print(f"Overall R-squared (R2) Score:     {overall_r2:.4f}")
    
    targets = ["x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW"]
    print("\nMetrics per Target:")
    for i, target in enumerate(targets):
        target_mse = mean_squared_error(y_test.numpy()[:, i], y_pred[:, i])
        target_r2 = r2_score(y_test.numpy()[:, i], y_pred[:, i])
        print(f"  {target}: R2 = {target_r2:.4f}, MSE = {target_mse:.4f}")

    print("\n--- 4. Saving the Model ---")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ann_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_ann()
