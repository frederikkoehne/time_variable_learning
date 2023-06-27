
import os
import torch
from torch import nn
from torchvision import datasets, transforms

import experiments

root_directory = '~/mldata' ## change this to your local root_directory for torchvision datasets

def mnist_exp_wrapper(n_epochs, batch_size, lr=1e-2, use_fmnist=False, number_of_blocks=3, **kwargs):
    """
    Create an experiment for the ResNet architecture using the MNIST dataset.

    Args:
        n_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimization.
        use_fmnist (bool): Flag indicating whether to use the FashionMNIST dataset instead of MNIST.
        number_of_blocks (int): Number of Tau blocks for the experiment.
        **kwargs: Additional keyword arguments for experiment configuration.

    Returns:
        experiments.Experiment: Initialized experiment object.
    """
    dataset = datasets.FashionMNIST if use_fmnist else datasets.MNIST
    data = dataset(root=root_directory, train=True,
                   transform=transforms.ToTensor())
    dataset = datasets.FashionMNIST if use_fmnist else datasets.MNIST
    data_test = dataset(root=root_directory, train=False,
                        transform=transforms.ToTensor())

    experiment = experiments.DenseResnetExperiment(
        preprocessing_layer=nn.Flatten(),
        loss_fn=nn.CrossEntropyLoss(),
        in_dimension=784,
        out_dimension=10,
        activation=nn.ReLU(),
        number_of_blocks=number_of_blocks,
        use_tau_for_first_layer=True,
        **kwargs
    )

    experiment.init_data(data=data, batch_size=batch_size, n_epochs=n_epochs)
    experiment.init_data(data=data_test, train=False)
    experiment.init_optimizer(torch.optim.SGD, lr=lr)

    return experiment


def mnist_frac_dnn_exp_wrapper(n_epochs, batch_size, lr=1e-2, use_fmnist=False, number_of_blocks=3, **kwargs):
    """
    Create an experiment for the FracDNN architecture using the MNIST dataset.

    Args:
        n_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimization.
        use_fmnist (bool): Flag indicating whether to use the FashionMNIST dataset instead of MNIST.
        number_of_blocks (int): Number of Tau blocks for the experiment.
        **kwargs: Additional keyword arguments for experiment configuration.

    Returns:
        experiments.Experiment: Initialized experiment object.
    """

    dataset = datasets.FashionMNIST if use_fmnist else datasets.MNIST
    data = dataset(root=root_directory, train=True,
                   transform=transforms.ToTensor())
    dataset = datasets.FashionMNIST if use_fmnist else datasets.MNIST

    data_test = dataset(root=root_directory, train=False,
                        transform=transforms.ToTensor())

    experiment = experiments.DenseFracDNNExperiment(
        preprocessing_layer=nn.Flatten(),
        loss_fn=nn.CrossEntropyLoss(),
        in_dimension=784,
        out_dimension=10,
        activation=nn.ReLU(),
        number_of_blocks=number_of_blocks,
        **kwargs
    )

    experiment.init_data(data=data, batch_size=batch_size, n_epochs=n_epochs)
    experiment.init_data(data=data_test, train=False)
    experiment.init_optimizer(torch.optim.SGD, lr=lr)

    return experiment


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as e:
            raise OSError('Error when creating directory') from e
