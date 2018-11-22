# Hyperspherical Variational Auto-Encoders
### Pytorch implementation of Hyperspherical Variational Auto-Encoders

## Overview
This library contains a Pytorch implementation of the hyperspherical variational auto-encoder, or S-VAE, as presented in [[1]](#citation)(http://arxiv.org/abs/1804.00891). Check also our blogpost (https://nicola-decao.github.io/s-vae).

* Don't use Pytorch? Take a look [here](https://github.com/nicola-decao/s-vae-tf) for a **tensorflow** implementation!

## Dependencies

* **python>=3.6**
* **pytorch>=0.4.1**: https://pytorch.org
* **scipy**: https://scipy.org
* **numpy**: https://www.numpy.org

## Installation

To install, run

```bash
$ python setup.py install
```

## Structure
* [distributions](https://github.com/nicola-decao/s-vae-pytorch/tree/master/hyperspherical_vae/distributions): Pytorch implementation of the von Mises-Fisher and hyperspherical Uniform distributions. Both inherit from `torch.distributions.Distribution`.
* [ops](https://github.com/nicola-decao/s-vae-pytorch/tree/master/hyperspherical_vae/ops): Low-level operations used for computing the exponentially scaled modified Bessel function of the first kind and its derivative.
* [examples](https://github.com/nicola-decao/s-vae-pytorch/tree/master/examples): Example code for using the library within a PyTorch project.

## Usage
Please have a look into the [examples folder](https://github.com/nicola-decao/s-vae-pytorch/tree/master/examples). We adapted our implementation to follow the structure of the [Pytorch probability distributions](https://pytorch.org/docs/stable/distributions.html).

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Sampling von Mises-Fisher
To sample the von Mises-Fisher distribution we follow the rejection sampling procedure as outlined by [Ulrich, 1984](http://www.jstor.org/stable/2347441?seq=1#page_scan_tab_contents). This simulation pipeline is visualized below:

<p align="center">
  <img src="https://i.imgur.com/aK1ze0z.png" alt="blog toy1"/>
</p>

_Note that as ![](http://latex.codecogs.com/svg.latex?%5Comega) is a scalar, this approach does not suffer from the curse of dimensionality. For the final transformation, ![](http://latex.codecogs.com/svg.latex?U%28%5Cmathbf%7Bz%7D%27%3B%5Cmu%29), a [Householder reflection](https://en.wikipedia.org/wiki/Householder_transformation) is utilized._

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com) or [Tim Davidson](mailto:itimrd@gmail.com).

## License
MIT

## Citation
```
[1] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T.,
and Tomczak, J. M. (2018). Hyperspherical Variational
Auto-Encoders. 34th Conference on Uncertainty in Artificial Intelligence (UAI-18).
```

BibTeX format:
```
@article{s-vae18,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R. and
          Falorsi, Luca and
          De Cao, Nicola and
          Kipf, Thomas and
          Tomczak, Jakub M.},
  journal={34th Conference on Uncertainty in Artificial Intelligence (UAI-18)},
  year={2018}
}
```
