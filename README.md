# Autoencoder-Frame-Prediction

Toy CNN autoencoder frame prediction using data from the [Berkeley Deep Drive Dataset](https://bdd-data.berkeley.edu/).

Built using TensorFlow 2.3. Requires natsort: `python -m pip install natsort`.

## Training

- Trained on 3713 videos from the dataset
- Videos were decoded to jpegs at 5 frames a second, resized to (180, 320, 3)
- Data separated by seconds (i.e. five images per each partition)
- Jpegs concatenated channel wise (shape=(180, 320, 12)) before being fed to the network, with label being fifth unconcatenated image
- MSE, Adam optimizer with learning rate of 1e-4, 6 epochs
- Final loss of `0.004605825524777174524777174`
