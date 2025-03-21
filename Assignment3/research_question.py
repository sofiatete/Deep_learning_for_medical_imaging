import argparse
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from train_VarNet import cli_main, build_args  # Import build_args from train_VarNet.py
from fastmri.data.mri_data import fetch_dir



def run_experiment(mask_type, center_fractions, accelerations, experiment_name="Mask_Test_Experiment"):
    # Set up ArgumentParser and define the required arguments
    parser = argparse.ArgumentParser()

    # Defining the arguments
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--mask_type", type=str, default=mask_type, help="Type of k-space mask")
    parser.add_argument("--center_fractions", type=float, nargs='+', default=center_fractions, help="Center fractions for the mask")
    parser.add_argument("--accelerations", type=int, nargs='+', default=accelerations, help="Acceleration factors for the mask")
    parser.add_argument("--experiment_name", type=str, default=experiment_name, help="Name of the experiment")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--data_path", type=str, default="FastMRIdata/", help="Path to the MRI data")
    # parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    # parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data for validation")
    # parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--challenge", type=str, default="multicoil", help="Challenge type (multicoil, etc.)")
    # parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader") 
    parser.add_argument("--test_path", type=str, default=None, help="Path to the test data")
    parser.add_argument("--accelerator", type=str, default=None, help="Accelerator type (ddp, etc.)")


    # Parse the arguments
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser = VarNetModule.add_model_specific_args(parser)
    args = parser.parse_args()

    # Run the training with the parsed arguments
    cli_main(args)



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
    # Example 1: Run with Gaussian mask
    run_experiment1(
        mask_type="gaussian", 
        center_fractions=[0.04], 
        accelerations=[4], 
        experiment_name="Gaussian_Mask_Test"
    )

    # Example 2: Run with Random mask
    run_experiment1(
        mask_type="random", 
        center_fractions=[0.1], 
        accelerations=[2], 
        experiment_name="Random_Mask_Test"
    )
