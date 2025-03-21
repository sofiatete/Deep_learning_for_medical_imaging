import argparse
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from train_VarNet import cli_main, build_args  # Import build_args from train_VarNet.py
from fastmri.data.mri_data import fetch_dir


def run_experiment(mask_type, center_fractions, accelerations, experiment_name="Mask_Test_Experiment"):
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
    run_experiment(
        mask_type="random", 
        center_fractions=[0.1], 
        accelerations=[2], 
        experiment_name="Random_Mask_Test"
    )