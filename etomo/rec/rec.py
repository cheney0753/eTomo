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

#=================
# the geometries class to glue the astra geometry and odl geometry
#=================
class Geometries:
    
    def __init__(self, astra_vol_geom, astra_proj_geom):
        '''
        The class geometries includes the astra volume geometry and the astra projection geometry for the HAADF projections.
        
        Parameters:
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
    
    def __init__(self,  data, geometries, norm = 'L2', regularization = 'none', name = []):
        """
        Define a tomographic reconstruction problem with optional regularization as following
        ```
            x = argmin_x sum_e norm( p - W * x ) + lambda * regularization(x)
        ```
        The regularization term can total variation TV(x).
        
        Parameters
        ----------
        geometries: object of the geometries 
            The geometries defining the forward projection model
        data: numpy.ndarray
            shape: P*M*N where N is the row number, self.data[:,:,m] is one sinogram
        norm: 'L2' or 'KL' 
            Defining the norm of data fidelity term
            'L2': the objective function is the l2 norm of the projection errors:  || p - W * x ||_2^2
            'KL': the objective function is Kullback-Leible divergence. May be useful for data with Poisson noise
            
        regularization: 'TV' or 'none'
            Regularization method.
            
        Reference
        -----------
        
        Hohage, T., & Werner, F. (2016).
        Inverse problems with Poisson data: statistical regularization theory, applications and algorithms. 
        Inverse Problems, 32(9), 93001. http://doi.org/10.1088/0266-5611/32/9/093001
        """
        
        if norm not in ['L2', 'KL']:
            raise Exception('The norm can only be L2 or KL')
        
        if regularization not in ['TV', 'TGV','SIRT', 'EM', 'TVR-DART', 'none']:
            raise Exception('The regularization term can only be TV or none')

        self.geometries = geometries
        
        # Note the data dimension order in odl is [num_angles,  num_row, num_col,]
        if self.geometries.dim == 3:
            data =  data.swapaxes(0,2).swapaxes(0, 1)
                    
        self.__data_odl = self.geometries.ray_trafo.range.element(data.copy(), dtype = np.float32)    
        
        self.ray_trafo = self.geometries.ray_trafo

        self.norm = norm
        
        self.regularization = regularization
        
        if not name:
          name = ['projection_data']
          
        self.name = name
    
    def normalize(self, std_ = 0):
        """
        normalize the data
        """
        
        if std_ == 0:
            std_ = np.std(self.__data_odl.data)
        

        self.__data_odl = self.__data_odl /std_
        
        
        return std_
    
    @property
    def data_odl(self):
        '''
        return the data

        '''
        return self.__data_odl
    
    @data_odl.setter
    def data_odl(self, data):
        """
        setter to set the data from a numpy array
                
        Parameters:
        ------
        data: numpy.array
            shape: P*M*N where N is the row number, self.data[:,:,m] is one sinogram
        
        Example:
        ------
            ```
                data_np = np.zeros( 1)
                proj = projection_data(...)
                proj.data_odl = data_np
            ```
        """
        self.__data_odl = self.geometries.ray_trafo.range.element(data.copy(), dtype = np.float32)    
        
        
def image_bin(img, size_bin):
    """
    function to bin a 2d image
    
    Args:        
        img: numpy.array
            image to be binned
        size_bin: int
            size of binning
    Return:        
        img_new: numpy.array
            binned image
    """
    if  not isinstance(size_bin, int):
        raise TypeError('size_bin must be int.')
    
    if img.shape[-1] % size_bin == 0:        
        return img.reshape(-1,  img.shape[1]//size_bin, size_bin, img.shape[2]//size_bin, size_bin).sum(axis = (2, 4))
    else:
        raise ValueError('size_bin is not correct.')