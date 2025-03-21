"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule
import time

import wandb
from pytorch_lightning.loggers import WandbLogger

def cli_main(args):
    pl.seed_everything(args.seed)

    # Wandb
    logger = WandbLogger(name=args.experiment_name, project='Exercise3')

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    print(f"Using {args.mask_type} mask...")
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False) # uses random masks
    val_transform = VarNetDataTransform(mask_func=mask) # uses fixed masks
    test_transform = VarNetDataTransform() # no masks
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # print how large the training/validation set is
    train_set = data_module.train_dataloader().dataset
    print("Size of trainingset:", len(train_set))
    val_set = data_module.val_dataloader().dataset
    print("Size of validationset:", len(val_set))

    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades, # number of iterations (data consistency layer + U-Net reconstruction layer)
        pools=args.pools, # number of pooling layers for U-Net
        chans=args.chans, # number of channels in the U-Net
        sens_pools=args.sens_pools, 
        sens_chans=args.sens_chans,  
        lr=args.lr, # learning rate
        lr_step_size=args.lr_step_size, # epoch interval at which to decrease learning rate
        lr_gamma=args.lr_gamma, # extent to which to decrease learning rate
        weight_decay=args.weight_decay, # weight regularization
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        print("Start training")
        start = time.time()
        trainer.fit(model, datamodule=data_module)
        end = time.time()
        print('Training time:', end-start)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set path to logs and saved model
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"
    data_path = "FastMRIdata/"

    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=(
            "random",
            "equispaced",
            "equispaced_fraction",
            "magic",
            "magic_fraction",
            "gaussian",
            "radial",
        ),
        default="random",
        type=str,
        help="Type of k-space mask",
    )

    # how much of the center to keep from subsampling
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--experiment_name",
        default='Masks_Comparison',
        type=str,
        help="Name of Experiment in WandB",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    args = parser.parse_args()

    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type=args.mask_type,
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    args = parser.parse_args()
    parser.set_defaults(
        num_cascades=2,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=args.learning_rate,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=args.num_epochs,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    # saves the best model based on validation loss
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    return args

def run_training(
    mask_type="random",
    center_fractions=[0.08],
    accelerations=[4],
    learning_rate=0.001,
    num_epochs=10,
    batch_size=1,
    experiment_name="Default_Experiment",
    data_path="FastMRIdata/",
    gpus=1,
    test_path=None,
    test_split=0.2,  
    sample_rate=1.0,  
):
    from argparse import Namespace

    args = Namespace(
        mode="train",
        mask_type=mask_type,
        center_fractions=center_fractions,
        accelerations=accelerations,
        experiment_name=experiment_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        data_path=data_path,
        test_path=test_path,
        test_split=test_split, 
        sample_rate=sample_rate,  
        challenge="multicoil",
        gpus=gpus,
        seed=42,
        deterministic=True,
        default_root_dir=f"checkpoints/{experiment_name}",
    )

    # Checkpointing
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"checkpoints/{experiment_name}",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # Model Hyperparameters
    args.lr = learning_rate  # ✅ MATCHED WITH cli_main
    args.lr_step_size = 40
    args.lr_gamma = 0.1
    args.weight_decay = 0.0
    args.num_cascades = 2
    args.pools = 4
    args.chans = 18
    args.sens_pools = 4
    args.sens_chans = 8

    # Data Loader Parameters
    args.num_workers = 4
    args.replace_sampler_ddp = False

    # Run the CLI training logic
    cli_main(args)



def run_cli():
    args = build_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    wandb.login()
    run_cli()
