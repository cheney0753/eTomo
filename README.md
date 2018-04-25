# astra-et
Electron tomography toolbox based on the ASTRA Tomography Toolbox 

The astra-et is a python toolbox made for electron tomography 
The basic forward and backward projection operations are based on the GPU-accelerated ASTRA-toolbox
http://www.astra-toolbox.com/
This toolbox provides tools and reconstruction algorithms frequently used tools for electron tomography, mainly for materials science.

## Installations
### Requirements/Dependency
*** Numpy for data processing
*** matplotlib for data visualization
*** The astra-toolbox with GPU support. For installation of astra-toolbox, refer to http://www.astra-toolbox.com/
*** Python environment: 64-bit 3.5 or 3.6
*** Cuda 5.5 or hihger
### Optional packages (might be used in the example codes)
*** HypersPy: a library to process EDS/EELS spectroscopic images
*** Operator Discretization Library (ODL): an operator-based package for optimization methods
### Conda:
TODO

### pip (for development)
Download the package, change to the source folder, type
>>> pip install -e .
in the terminal

## Functions:
### Data I/O:
Support reading .mrc files. .mrc is a standard file format for tomographic measurement data from electron microscopes.

### Preprocessing:
Basic functions are provided:
Normalization:
Alignment:

### Data visualization
A slice viewer is provided basd on matplotlib. It is useful for viewing 3D volume / tilt-series of projection images
Support viewing and ploting the reconstruction while the iterative reconstruction algorithm is running

### Reconstruction algorithms
Default reconstruction algorithms implemented as follows:
*** Analytical algorithms: FBP (simple and fast)
*** Numerical algorithms: SIRT (smooth reconstructions)
*** Statistical algorithms: EM (for data with strong Poisson noise)
*** Advanced primal-dual algorithms: Chambolle-Pock 
It is also possible to implement your own reconstruction algorithm

### Post-processing
TODO

## Examples:
Examples 
*** TV-regularized STEM tomographic reconstruction 
*** Customized reconstruction algorithm
*** HAADF-EDS bimodal tomographic reconstruction
*** Nonlinearity correction for STEM tomography
...

## Acknowledgment:
This software is a product of the computational imaging group at CWI, Amsterdam. 
https://www.cwi.nl/research/groups/computational-imaging



