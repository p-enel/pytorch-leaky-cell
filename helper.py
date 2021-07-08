import torch


def random_weights(input_size: int = 1,
                   output_size: int = None,
                   sparsity: float = 1,
                   mask: torch.tensor = None,
                   distribution: str = 'normal',
                   dist_params: [float, float] = [0, 1],
                   spectral_radius: float = None,
                   seed: float = None):
    '''
    Generate random weights according to parameters

    Arguments:
    ----------
    input_size: int - the number of input units
    output_size: int - the number of output units
    sparsity: float in interval [0, 1] - the sparsity of the wright matrix
    mask: torch.tensor of boolean - a mask applied to the weight matrix
    distribution: str among ['normal', 'uniform'] - the type of weight distribution
    dist_params: [mean, std] for normal distrib
                 [min, max] for uniform
    spectral_radius: float in interval [0, inf) - the highest absolute eigen value of the matrix
    seed: float - the seed used to pseudo-randomly generate the weights for reproducibility
    '''
    if input_size is None:
        output_size = input_size

    if spectral_radius is not None and input_size != output_size:
        raise(ValueError, 'spectral_radius can be defined only for square matrices (nbUnitsIN == nbUnitsOUT)')

    if sparsity > 1 or sparsity < 0:
        raise(ValueError, 'sparsity argument is a float between 0 and 1')

    # Set the seed for the random weight generation
    if seed is not None:
        torch.manual_seed(seed)

    # Uniform random distribution of weights:
    if distribution == 'uniform':
        if dist_params is not None:
            minimum, maximum = dist_params
        else:
            minimum, maximum = [-1, 1]
        weights = (torch.rand((output_size, input_size)) * (maximum - minimum) + minimum)

    # Normal (gaussian) random distribution of weights:
    elif distribution == 'normal':
        if dist_params is not None:
            mu, sigma = dist_params
        else:
            mu, sigma = [0, 1]
        weights = torch.randn(output_size, input_size) * sigma + mu

    weights = weights * (torch.rand_like(weights) < sparsity)

    if mask is not None:
        weights = weights * mask

    if spectral_radius is not None:
        currentSpecRad = max(torch.linalg.norm(torch.eig(weights)[0], ord=2, dim=1))
        weights = weights / currentSpecRad * spectral_radius

    return weights
