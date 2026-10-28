# Checkpoints Directory

This directory is used for intermediate training checkpoints and evaluation artifacts:
- `checkpoint_epoch_*.pth`: Epoch checkpoints during training runs.
- `training_losses.csv`: Training and validation loss progression.
- `loss_curves.png`: Training curve plots for Generator, Discriminator, and L1 reconstruction loss.

Trained weights intended for application inference should be placed in `models/generator.pth`.
