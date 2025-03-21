import time
from argparse import ArgumentParser
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path


def train_with_config():
    pl.seed_everything(42)

    logger = WandbLogger(name=experiment_name, project="Mask_Comparison")

    # Create mask
    mask = create_mask_for_mask_type(mask_type, center_fractions, accelerations)
    
    # Data transforms
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)

    data_path_obj = Path(data_path)
    # Data module
    data_module = FastMriDataModule(
        data_path=data_path_obj,
        challenge="multicoil",
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=None,
        batch_size=batch_size,
        num_workers=4,
    )

    # Model initialization
    model = VarNetModule(
        num_cascades=num_cascades,
        pools=pools,
        chans=chans,
        sens_pools=sens_pools,
        sens_chans=sens_chans,
        lr=learning_rate,
    )

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        default_root_dir=f"checkpoints/{experiment_name}",
    )

    # Training
    print(f"Starting training for mask type: {mask_type}")
    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # Mask parameters
    mask_type = "gaussian"  # Options: random, equispaced, equispaced_fraction, radial, gaussian
    center_fractions = [0.08]  # Fraction of k-space center to keep
    accelerations = [4]  # Acceleration factor

    # Model parameters
    num_cascades = 2
    pools = 4
    chans = 18
    sens_pools = 4
    sens_chans = 8

    # Training parameters
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 1

    # Paths
    data_path = "FastMRIdata/"
    experiment_name = "Gaussian_Mask_Test"
    train_with_config()
