
import math
import torch


class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim
    
    def __init__(self, dim, validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim

    def sample(self, shape=torch.Size()):        
        output = torch.distributions.Normal(0, 1).sample(
            (shape if isinstance(shape, torch.Size) else torch.Size([shape])) + torch.Size([self._dim + 1]))

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()
    
    def log_prob(self, x):
        return - torch.ones(x.shape[:-1]) * self.__log_surface_area()

    def __log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
            torch.Tensor([(self._dim + 1) / 2]))
        