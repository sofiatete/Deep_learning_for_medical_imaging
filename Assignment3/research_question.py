
# Import the run_training function from train_VarNet.py
from train_VarNet import run_training

# Example 1: Running with Gaussian mask and specific center fractions and accelerations
run_training(
    mask_type="gaussian",          # Type of k-space mask
    center_fractions=[0.04],      # Center fraction for k-space
    accelerations=[4],            # Acceleration factor
    learning_rate=0.001,          # Learning rate for optimizer
    num_epochs=15,                # Number of epochs
    batch_size=1,                 # Batch size
    experiment_name="Gaussian_Mask_Test", # Experiment name for logging
    data_path="FastMRIdata/",     # Path to the MRI data
    gpus=1,                       # Number of GPUs to use
    test_split=0.2,               # Fraction of data to use for validation
    sample_rate=1.0,              # Sample rate (usually 1.0)
)

# Example 2: Running with Random mask and different center fractions and accelerations
run_training(
    mask_type="random",           # Type of k-space mask
    center_fractions=[0.1],       # Center fraction for k-space
    accelerations=[2],            # Acceleration factor
    learning_rate=0.0005,         # Learning rate for optimizer
    num_epochs=10,                # Number of epochs
    batch_size=1,                 # Batch size
    experiment_name="Random_Mask_Test", # Experiment name for logging
    data_path="FastMRIdata/",     # Path to the MRI data
    gpus=1,                       # Number of GPUs to use
    test_split=0.2,               # Fraction of data to use for validation
    sample_rate=1.0,              # Sample rate
)
