from research_question import run_training

# Experiment 1: Gaussian mask
run_training(
    mask_type="gaussian",
    center_fractions=[0.04],
    accelerations=[8],
    learning_rate=0.0005,
    num_epochs=15,
    batch_size=2,
    experiment_name="Gaussian_Mask_Test",
)

# Experiment 2: Random mask
run_training(
    mask_type="random",
    center_fractions=[0.08],
    accelerations=[4],
    learning_rate=0.001,
    num_epochs=10,
    batch_size=1,
    experiment_name="Random_Mask_Test",
)
