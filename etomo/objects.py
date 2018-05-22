#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:41:20 2018

@author: zhong

etomo.rec module for the etomo package
"""

import odl
import astra
import numpy as np

from etomo.utils import image_bin

#=================
# the geometries class to glue the astra geometry and odl geometry
#=================
class Geometries:
    
    def __init__(self, astra_vol_geom, astra_proj_geom):
        '''
        The class geometries includes the astra volume geometry and the astra projection geometry for the HAADF projections.
        
        Args:
        -----------
        
        astra_vol_geom : astra volume geometry dictionary
            support 2D or 3D geometry
        astra_proj_geom : astra projection geometry dictionary
            The projection geometry defining the projection geometry
        '''
        
        # Dimension
        self.dim = len(astra_vol_geom) - 1
        
        if self.dim != 2 and self.dim != 3:
            raise ValueError("The volume dimension must be 2D or 3D.")
        else:
            self.vol_geom = astra_vol_geom

        if self.dim == 2 and astra_proj_geom['type'] == 'parallel3d':
            raise Exception("The projection dimension type must be 'parallel'")
        elif self.dim == 3 and astra_proj_geom['type'] == 'parallel' :
            raise Exception("The projection dimension type must be 'parallel3d'")
        else:    
            self.proj_geom  = astra_proj_geom

        # Add tiny difference between the same angles, which are not supported by odl
        dif = np.diff(astra_proj_geom['ProjectionAngles'])
        cond = (dif==0)
        cond = np.append(cond, [False])
#        print(cond.shape)
        for i, tilt in enumerate(astra_proj_geom['ProjectionAngles']):
            if cond[i]:
                astra_proj_geom['ProjectionAngles'][i]-=1e-5
                
        angle_partition = odl.nonuniform_partition(astra_proj_geom['ProjectionAngles'])
        
        # set up the 2d geometry in odl
        if self.dim == 2 :
            # Make a parallel beam geometry with flat detector            
            detector_count = astra_proj_geom['DetectorCount']
            d_width = astra_proj_geom['DetectorWidth']
            detector_partition = odl.uniform_partition(-detector_count*d_width/2, detector_count*d_width/2, detector_count)
            self.geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
            
            # Set up the 2d volume geometry (space) in odl
            shape = [astra_vol_geom['GridRowCount'], astra_vol_geom['GridColCount'] ]
            max_pt = [shape[0]/2, shape[1]/2]
            min_pt = [-shape[0]/2, -shape[1]/2]
            self.space = odl.uniform_discr(min_pt, max_pt, shape, dtype = 'float32')

        elif self.dim == 3:
            # Make a parallel beam geometry with flat detector            
            row_count = astra_proj_geom['DetectorRowCount']
            col_count = astra_proj_geom['DetectorColCount']
            x_width = astra_proj_geom['DetectorSpacingX']
            y_width = astra_proj_geom['DetectorSpacingY']
            detector_partition = odl.uniform_partition([-row_count*x_width/2, -col_count*y_width/2], [row_count*x_width/2, col_count*y_width/2]
                                                       , [row_count, col_count])
            
            self.geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition, axis=[1, 0, 0])
            
            # Set up the 3d volume geometry (space) in odl
            shape = [astra_vol_geom['GridRowCount'], astra_vol_geom['GridColCount'], astra_vol_geom['GridSliceCount'] ]
            max_pt = [shape[0]/2, shape[1]/2, shape[2]/2]
            min_pt = [-shape[0]/2, -shape[1]/2, -shape[2]/2]
            self.space = odl.uniform_discr(min_pt, max_pt, shape, dtype = 'float32')

        self.ray_trafo = odl.tomo.RayTransform(self.space, self.geometry, impl='astra_cuda')


#=================
# the projection data class to glue odl and astra
#================
class Projection_data:
    
    def __init__(self, data, geometries):
        """
        Define a tomographic reconstruction problem with optional regularization as following
        ```
            x = argmin_x sum_e norm( p - W * x ) + lambda * regularization(x)
        ```
        The regularization term can total variation TV(x).
        
        Args:
        ----------
        geometries : object of the geometries 
            The geometries defining the forward projection model
        data : numpy.ndarray
            shape: P*M*N where N is the row number, self.data[:,:,m] is one sinogram
        norm : 'L2' or 'KL' 
            Defining the norm of data fidelity term
            'L2': the objective function is the l2 norm of the projection errors:  || p - W * x ||_2^2
            'KL': the objective function is Kullback-Leible divergence. May be useful for data with Poisson noise
            
        Reference
        -----------
        
        Hohage, T., & Werner, F. (2016).
        Inverse problems with Poisson data: statistical regularization theory, applications and algorithms. 
        Inverse Problems, 32(9), 93001. http://doi.org/10.1088/0266-5611/32/9/093001
        """
        
        
        # if regularization not in ['TV', 'TGV','SIRT', 'EM', 'TVR-DART', 'none']:
            # raise Exception('The regularization term can only be TV or none')

        self.geometries = geometries
        
        # Note the data dimension order in odl is [num_angles,  num_row, num_col,]
        if self.geometries.dim == 3:
            # data =  data.swapaxes(0,2).swapaxes(0, 1)
            data = np.transpose(data, (1, 2, 0))
        
        self.__data_odl = self.geometries.ray_trafo.range.element(data.copy(), dtype = np.float32)    

    def normalize(self, std_ = 0):
        """ normalize the data, divided by the standard deviation."""
        if std_ == 0:
            std_ = np.std(self.__data_odl.asarray())

        self.__data_odl = self.__data_odl /std_
        return std_
    
    @property
    def data_odl(self):
        '''
        return the data in the odl format

        '''
        return self.__data_odl

    @data_odl.setter
    def data_odl(self, odl_data):
        '''
        setter to set the data
        '''
        assert self.__data_odl.space  == odl_data.space
        self.__data_odl = odl_data

    @property
    def data(self):
        '''return the data in the numpy array'''
        return self.__data_odl.asarray

    @data.setter
    def data(self, data):
        """
        setter to set the data from a numpy array
                
        Parameters:
        ------
        data: numpy.array
            shape: P*M*N where N is the row number, self.data[:,:,m] is one sinogram
        
        Example:
        ________ 
            ```
                data_np = np.zeros( 1)
                proj = projection_data(...)
                proj.data_odl = data_np
            ```
        """
        self.__data_odl = self.geometries.ray_trafo.range.element(data.copy(), dtype = np.float32)

    def reconstruct(self, norm, regularization, alg = 'PDHG', n_iter = 20, lmbd = 0.0, nonnegativity = True, callback = None,  **kwargs):
        """
        Reconstruct from the projdata.
        Args :
        _____________

        alg : string
            the numerical algorithm, one of ('SIRT', 'EM', 'PDHG'). PDHG is the chamboll-pock algorithm 
        """
        if not regularization:
            if alg not in ('PDHG', 'DR'):
                print('Only the PDHG or DR algorithm is available for TV. Switch to the PDHG algorithm.')
                alg = 'PDHG'
                
        if regularization is 'TV':
            x = rec_odl(self.__data_odl, self.geometries.ray_trafo, norm, n_iter, lmbd, nonnegativity, algorithm = alg, callback = callback)

        elif regularization is 'TNV':
            try:
                rec_sub = kwargs['rec_sub']
            except KeyError:
                raise KeyError('rec_sub must be defined as a keyword argument.')

            try:
                rec_sub_odl = self.geometries.ray_trafo.domain.element(rec_sub)
            except:
                raise Exception('rec_sub must have the same data shape.')
            x = rec_joined_rec(self.__data_odl, self.geometries.ray_trafo, rec_sub_odl, norm, n_iter, lmbd, nonnegativity, regularization, algorithm = alg, callback = callback)

    #TODO: implement the SIRT and EM algorithm, implement the TNV joint regularization
        return x.asarray()

def rec_odl(data_odl, ray_trafo, norm = 'L2', n_iter = 100, lmbd = 0.0, nonnegativity = True, regularization = 'TV', algorithm = 'PDHG', callback = None):
    """
    tomographic reconstruction pdhg algorithm defined for the projection_data class

    Args :
    _______
    
    """
    assert norm in ('KL', 'L2')
    assert regularization in ('TV', None, 'None', 'none')
    assert algorithm in ('PDHG', 'DR')

    # define the ray transform
    domain = ray_trafo.domain
    # Define the gradient operator
    grad = odl.Gradient(domain)

    # Create box constraint functional for the channel data
    if nonnegativity:
        f = odl.solvers.IndicatorBox(domain, lower = 0 )
    else:
        f = odl.solvers.IndicatorBox(domain)

    # Create data discrepancy functionals for the sum data
    if norm is 'KL':
        g_data = odl.solvers.KullbackLeibler(ray_trafo.range, prior= data_odl)
    elif norm is 'L2':
        g_data = odl.solvers.L2NormSquared(ray_trafo.range).translated(data_odl)

    if regularization is 'TV':
        g_reg = lmbd * odl.solvers.GroupL1Norm(grad.range)    
    elif not regularization:
        g_reg = None
    else:
        raise Exception('Regularization method not defined.')
    # print( regularization, lmbd, algorithm)
    
    x = domain.zero()
    if algorithm is 'PDHG': 
        if not regularization:
            g = g_data 
            lin_ops = ray_trafo
        elif regularization is 'TV':
            g = odl.solvers.SeparableSum(g_data, g_reg)
            lin_ops = odl.BroadcastOperator(ray_trafo, grad)
            
        op_norm = 1.1 * odl.power_method_opnorm(lin_ops, maxiter=4)
        tau = sigma = 1.0 / op_norm
        if odl.__version__ >= '0.6.1':
            pdhg_solver = odl.solvers.pdhg
        else:
            pdhg_solver = odl.solvers.chambolle_pock_solver
        pdhg_solver(x, g, f, lin_ops, tau=tau, sigma=sigma, niter=n_iter, callback = callback) 
        
    elif algorithm is 'DR':
        if not regularization:
            g = g_data 
            lin_ops = ray_trafo
        elif regularization is 'TV':
            g = [g_data, g_reg]
            lin_ops = [ray_trafo, grad]
        tau = 2.0 / len(lin_ops)
        sigma = [1 / odl.power_method_opnorm(op, rtol=0.1)**2 for op in lin_ops]
        odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau=tau, sigma=sigma, niter=n_iter, callback = callback)
    return x

def rec_joined_rec(data_odl, ray_trafo, rec_sup, norm = 'L2', n_iter = 100, lmbd = 0.0, nonnegativity = True, regularization = 'TNV', algorithm = 'PDHG', callback = None):
    """
    tomographic reconstruction pdhg algorithm defined for the projection_data class

    Args :
    _______
    
    """
    assert norm in ('KL', 'L2')
    assert regularization in ('TNV',)
    assert algorithm in ('PDHG', 'DR')
    try:
        assert rec_sup.space == ray_trafo.domain
    except:
        raise ValueError('The rec_sup does not have the same space as the reconstruction.')
    # define the ray transform
    domain = odl.ProductSpace( ray_trafo.domain, 2)
    # Define the gradient operator
    grad = odl.Gradient(ray_trafo.domain)
    
    # Create box constraint functional for the channel data
    if nonnegativity:
        f = odl.solvers.IndicatorBox(domain, lower = 0 )
    else:
        f = odl.solvers.IndicatorBox(domain)

    # Create data discrepancy functionals for the sum data
    if norm is 'KL':
        g_data = odl.solvers.KullbackLeibler(ray_trafo.range, prior= data_odl)
    elif norm is 'L2':
        g_data = odl.solvers.L2NormSquared(ray_trafo.range).translated(data_odl)
        
    # Create the zero indicator to map rec_sup into the reconstruction domain
    op_ident = odl.IdentityOperator(rec_sup.space)
    g_sup = odl.solvers.IndicatorZero(rec_sup.space).translated(rec_sup)
    
    if regularization is 'TNV':
        diag_grad = odl.DiagonalOperator(grad, grad)
        g_reg = lmbd * odl.solvers.NuclearNorm(diag_grad.range, singular_vector_exp = 1)
    else:
        raise Exception('Regularization method not defined.')

    g_sum = odl.solvers.SeparableSum(g_data, g_sup)
    op_diag = odl.DiagonalOperator(ray_trafo, op_ident) 

    x = domain.zero()
    if algorithm is 'PDHG': 
        raise Exception('PDHG for tnv regularization is not implemented yet, because odl.BroadcastOperator() does not support sub-operators. Use DR algorithm instead.') 
        # g = odl.solvers.SeparableSum(g_sum, g_reg)
        # lin_ops = odl.BroadcastOperator(op_diag, diag_grad)
        # op_norm = 1.1 * odl.power_method_opnorm(lin_ops, maxiter=4)
        # tau = sigma = 1.0 / op_norm

        # if odl.__version__ >= '0.6.1':
        #     pdhg_solver = odl.solvers.pdhg
        # else:
        #     pdhg_solver = odl.solvers.chambolle_pock_solver

        # pdhg_solver(x, g, f, lin_ops, tau=tau, sigma=sigma, niter=n_iter, callback = callback)
    elif algorithm is 'DR':

        g = [g_sum, g_reg]
        lin_ops = [op_diag, diag_grad]
        tau = 2.0 / len(lin_ops)
        sigma = [1 / odl.power_method_opnorm(op, rtol=0.1)**2 for op in lin_ops]
        odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau=tau, sigma=sigma, niter=n_iter, callback = callback)
    return x[0]
