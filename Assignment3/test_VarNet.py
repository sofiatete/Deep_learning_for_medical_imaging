"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
from evaluation_metrics import ssim, nmse, mse, psnr
import h5py
from typing import Tuple

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.transforms import center_crop

from fastmri.pl_modules import FastMriDataModule, VarNetModule

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations,
    )
    # mask = create_mask_for_mask_type(
    #     12, args.center_fractions, args.accelerations,
    # )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform(mask_func=mask)
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
    test_set = data_module.test_dataloader().dataset
    print("Size of trainingset:", len(test_set))


    # ------------
    # model
    # ------------

    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    return


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    # path_config = pathlib.Path("/content/gdrive/MyDrive/DL_4_MI/Assigment3/save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    # data_path = "/content/gdrive/MyDrive/DL_4_MI/Assigment3/FastMRIdata/"
    data_path = '/gpfs/work5/0/prjs1312/Recon_exercise/FastMRIdata/'

    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="test",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
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
        default=[1.5],
        type=int,
        help="Acceleration rates to use for masks",
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
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        # mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        mask_type="random",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,
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
        # strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TESTING
    # ---------------------
    cli_main(args)


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

# perform inverse fourier transform and shifting
def fourier_transform(kspace):
    """Reconstructs an image using the Fourier transform with proper shifting."""
    dim1, dim2 = -2, -1  # Last two dimensions (H, W)

    # Shift k-space center to middle before performing IFT
    kspace_shifted = np.fft.ifftshift(kspace, axes=(dim1, dim2))

    # Perform the inverse 2D Fourier Transform
    image = np.fft.ifft2(kspace_shifted, axes=(dim1, dim2))

    # Shift the final image to center it correctly
    image = np.fft.fftshift(image, axes=(dim1, dim2))

    return image



def evaluate_test_data_quantitatively(datapath, reconpath):
    #######################
    # Start YOUR CODE    #
    #######################
    # Load ground truth and reconstruction data once 
    ground_truth_files = sorted(pathlib.Path(datapath).glob('*.h5')) 
    reconstruction_files = sorted(pathlib.Path(reconpath).glob('*.h5'))

    mse_values = []
    nmse_values = []
    psnr_values = []
    ssim_values = []
    
    # Loop over each pair of ground truth and reconstructed images
    for gt_file, recon_file in zip(ground_truth_files, reconstruction_files):
        # Load ground truth and reconstruction images
        with h5py.File(gt_file, 'r') as f:
            gt = f['/kspace'][:]  # Assuming the ground truth is stored under '/kspace'
        
        with h5py.File(recon_file, 'r') as f:
            recon = f['/reconstruction'][:]  # Assuming the reconstruction is stored under '/reconstruction'
        
        # Center crop the ground truth image to match the size of the reconstructed image
        gt = np.squeeze(gt, axis=1) 
        gt = center_crop(gt, recon.shape[1:])

        # Apply Fourier transform (shifting and inverse shifting)
        gt = fourier_transform(gt)
        recon = fourier_transform(recon)

        # Assuming gt and recon are complex, apply magnitude or real part
        if np.iscomplexobj(gt):
            gt = np.abs(gt)  # Or np.real(gt) if you prefer to discard the imaginary part

        if np.iscomplexobj(recon):
            recon = np.abs(recon)  # Or np.real(recon)

        # Then you can proceed to calculate PSNR or other metrics
        psnr_val = psnr(gt, recon)

        
        # Compute metrics for the current image
        mse_val = mse(gt, recon)
        nmse_val = nmse(gt, recon)
        psnr_val = psnr(gt, recon)
        ssim_val = ssim(gt, recon)
        
        # Append the metrics to the lists
        mse_values.append(mse_val)
        nmse_values.append(nmse_val)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    # Calculate the mean (or median) of each metric across all images
    mean_mse = np.mean(mse_values)
    mean_nmse = np.mean(nmse_values)
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    # Print the results
    print(f"Mean MSE: {mean_mse}")
    print(f"Mean NMSE: {mean_nmse}")
    print(f"Mean PSNR: {mean_psnr}")
    print(f"Mean SSIM: {mean_ssim}")
    #######################
    # END OF YOUR CODE    #
    #######################
    return


def save_image(fig, output_path: str):
    """ Save the figure as a PNG file to the specified path. """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    fig.savefig(output_path, bbox_inches='tight', dpi=300)  
    plt.close(fig)  

def evaluate_test_data_qualitatively(datapath, reconpath, output_dir):
    #######################
    # Start YOUR CODE    #
    #######################
    
    # Get the list of all ground truth files and reconstruction files in the directory
    ground_truth_files = sorted(pathlib.Path(datapath).glob('*.h5'))
    reconstruction_files = sorted(pathlib.Path(reconpath).glob('*.h5'))

    # Loop over each pair of ground truth and reconstructed images
    for gt_file, recon_file in zip(ground_truth_files, reconstruction_files):
        print(f"Processing: {gt_file.name} and {recon_file.name}")
        
        # Load ground truth and reconstruction images
        with h5py.File(gt_file, 'r') as f:
            gt = f['/kspace'][:]  
        
        with h5py.File(recon_file, 'r') as f:
            recon = f['/reconstruction'][:]  

        # If complex-valued, squeeze and center crop
        gt = np.squeeze(gt, axis=1)  # Squeeze the singleton dimension
        gt = center_crop(gt, recon.shape[1:])
        
        if np.iscomplexobj(gt):
            gt_magnitude = np.abs(gt)
            gt_phase = np.angle(gt)
            gt_real = np.real(gt)
            gt_imag = np.imag(gt)
        else:
            gt_magnitude = gt
            gt_phase = np.zeros_like(gt)
            gt_real = gt
            gt_imag = np.zeros_like(gt)
        
        if np.iscomplexobj(recon):
            recon_magnitude = np.abs(recon)
            recon_phase = np.angle(recon)
            recon_real = np.real(recon)
            recon_imag = np.imag(recon)
        else:
            recon_magnitude = recon
            recon_phase = np.zeros_like(recon)
            recon_real = recon
            recon_imag = np.zeros_like(recon)
        
        # Apply Fourier transform (shifting and inverse shifting)
        gt_image = fourier_transform(gt)
        recon_image = fourier_transform(recon)

        # Take the magnitude to get real-valued images
        gt_image_magnitude = np.abs(gt_image).squeeze()  # Remove extra dimensions, if any
        recon_image_magnitude = recon_image.squeeze()

        # Select the center slice (index = middle index of the 3D data)
        center_index = gt_image_magnitude.shape[0] // 2  # Find the center slice index
        gt_image_magnitude = gt_image_magnitude[center_index]  # Take the center slice
        recon_image_magnitude = np.abs(recon_image_magnitude[center_index])  # Take the center slice

        fig, axs = plt.subplots(2, 4, figsize=(15, 8))

        # Ground truth magnitude (center slice)
        axs[0, 0].imshow(gt_image_magnitude, cmap='gray')
        axs[0, 0].set_title('Ground Truth Magnitude')

        # Ground truth phase
        axs[0, 1].imshow(gt_phase[center_index], cmap='gray')
        axs[0, 1].set_title('Ground Truth Phase')

        # Ground truth real part
        axs[0, 2].imshow(gt_real[center_index], cmap='gray')
        axs[0, 2].set_title('Ground Truth Real')

        # Ground truth imaginary part
        axs[0, 3].imshow(gt_imag[center_index], cmap='gray')
        axs[0, 3].set_title('Ground Truth Imaginary')

        # Reconstruction magnitude (center slice)
        axs[1, 0].imshow(recon_image_magnitude, cmap='gray')
        axs[1, 0].set_title('Reconstructed Magnitude')

        # Reconstruction phase
        axs[1, 1].imshow(recon_phase[center_index], cmap='gray')
        axs[1, 1].set_title('Reconstructed Phase')

        # Reconstruction real part
        axs[1, 2].imshow(recon_real[center_index], cmap='gray')
        axs[1, 2].set_title('Reconstructed Real')

        # Reconstruction imaginary part
        axs[1, 3].imshow(recon_imag[center_index], cmap='gray')
        axs[1, 3].set_title('Reconstructed Imaginary')

        output_path = os.path.join(output_dir, f"{gt_file.stem}_comparison.png")
        save_image(fig, output_path)
        print(f"Saved: {output_path}")
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return

def evaluate_test_data_qualitatively2(datapath, reconpath, output_dir):
    # Get the list of all ground truth files and reconstruction files in the directory
    ground_truth_files = sorted(pathlib.Path(datapath).glob('*.h5'))
    reconstruction_files = sorted(pathlib.Path(reconpath).glob('*.h5'))

    for gt_file, recon_file in zip(ground_truth_files, reconstruction_files):
        print(f"Processing: {gt_file.name} and {recon_file.name}")
        
        # Load ground truth and reconstruction images
        with h5py.File(gt_file, 'r') as f:
            gt_kspace = f['/kspace'][:]  # Load k-space
        
        with h5py.File(recon_file, 'r') as f:
            recon_image = f['/reconstruction'][:]  # Reconstructed image in image domain

        # Squeeze and center crop ground truth to match reconstruction dimensions
        gt_kspace = np.squeeze(gt_kspace, axis=1)  
        gt_kspace = center_crop(gt_kspace, recon_image.shape[1:])

        # Convert ground truth k-space to image domain
        gt_image = fourier_transform(gt_kspace)  # Apply Fourier transform

        # Get the magnitude correctly for visualization
        gt_magnitude = np.abs(gt_image)  # Corrected: Magnitude in image domain
        recon_magnitude = np.abs(recon_image)  # Ensure consistent magnitude handling

        # Get center slice index
        center_index = gt_magnitude.shape[0] // 2  # Center slice index

        # Extract center slice
        gt_magnitude_slice = gt_magnitude[center_index]  
        recon_magnitude_slice = recon_magnitude[center_index]

        # Get phase, real, and imaginary parts correctly
        gt_phase_slice = np.angle(gt_image[center_index])
        gt_real_slice = np.real(gt_image[center_index])
        gt_imag_slice = np.imag(gt_image[center_index])

        recon_phase_slice = np.angle(recon_image[center_index])
        recon_real_slice = np.real(recon_image[center_index])
        recon_imag_slice = np.imag(recon_image[center_index])

        # Plot results
        fig, axs = plt.subplots(2, 4, figsize=(15, 8))

        axs[0, 0].imshow(gt_magnitude_slice, cmap='gray')
        axs[0, 0].set_title('Ground Truth Magnitude')

        axs[0, 1].imshow(gt_phase_slice, cmap='gray')
        axs[0, 1].set_title('Ground Truth Phase')

        axs[0, 2].imshow(gt_real_slice, cmap='gray')
        axs[0, 2].set_title('Ground Truth Real')

        axs[0, 3].imshow(gt_imag_slice, cmap='gray')
        axs[0, 3].set_title('Ground Truth Imaginary')

        axs[1, 0].imshow(recon_magnitude_slice, cmap='gray')
        axs[1, 0].set_title('Reconstructed Magnitude')

        axs[1, 1].imshow(recon_phase_slice, cmap='gray')
        axs[1, 1].set_title('Reconstructed Phase')

        axs[1, 2].imshow(recon_real_slice, cmap='gray')
        axs[1, 2].set_title('Reconstructed Real')

        axs[1, 3].imshow(recon_imag_slice, cmap='gray')
        axs[1, 3].set_title('Reconstructed Imaginary')

        output_path = os.path.join(output_dir, f"{gt_file.stem}_comparison.png")
        save_image(fig, output_path)
        print(f"Saved: {output_path}")
    
    return


if __name__ == "__main__":
    # Run testing the network
    run_cli()
    datapath = '/gpfs/work5/0/prjs1312/Recon_exercise/FastMRIdata/multicoil_test/'
    reconpath = 'varnet/varnet_demo/reconstructions/'

    # Specify the output directory for saving images
    output_dir = './qualitative_results'

    # Quantitative evaluation
    # evaluate_test_data_quantitatively(datapath, reconpath)

    # Qualitative evaluation
    evaluate_test_data_qualitatively2(datapath, reconpath, output_dir)
