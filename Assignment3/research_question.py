import argparse
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from train_VarNet import cli_main, build_args  # Import build_args from train_VarNet.py
from fastmri.data.mri_data import fetch_dir


def run_experiment1(mask_type, center_fractions, accelerations, experiment_name="Mask_Test_Experiment"):
    # Get default arguments from train_VarNet.py
    args = build_args()  # This ensures all default arguments (including data_path) are set

    # Override only the necessary arguments
    args.mask_type = mask_type
    args.center_fractions = center_fractions
    args.accelerations = accelerations
    args.experiment_name = experiment_name

    # Run the training with the modified arguments
    cli_main(args)



if __name__ == "__main__":

    # Example 2: Run with Random mask
    run_experiment1(
        mask_type="random", 
        center_fractions=[0.1], 
        accelerations=[2], 
        experiment_name="Random_Mask_Test"
    )

    # Example 1: Run with Gaussian mask
    run_experiment1(
        mask_type="gaussian", 
        center_fractions=[0.04], 
        accelerations=[4], 
        experiment_name="Gaussian_Mask_Test"
    )
