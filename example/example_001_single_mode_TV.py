from etomo.objects import (Projection_data, Geometries)
from etomo.utils import (image_bin,)
import argparse
import os
import sys
import scipy.misc as scms
import scipy.io as scio
import numpy as np
import astra
if __name__ == '__main__':
    # TODO
    # difine the parser
    
    pars = argparse.ArgumentParser()
    
    pars.add_argument("--nit", type=int, default = 100, help = "The number of iterations.")
    pars.add_argument("--slice", type=int, default = -1, help = "The number of reconstructed slice. When the number is -1, the entire volume is reconstructed. Otherwise the slice indicated by the number is reconstructed.")
    pars.add_argument("--dim", type=int, default = 2, help = "The reconstruction dimension: 3D or 2D (slice by slice). 2: 2D; 3: 3D. Default is 2D.")
    pars.add_argument("--lmbd", type=float, default = 1e-1, help = "The regularization paratmeter.")
    pars.add_argument("--show", default = False, action ='store_true')
    
    args = pars.parse_args()
    nit = args.nit
    dim = str(args.dim)+'D'
    slc = args.slice
    lmbd = args.lmbd
    show = args.show
    
    # temp
    slc = 50
    show = False  
    if dim == '2D':
        try:
            assert(slc >= 0)
        except:
            raise ValueError('The slice number is not define.')
            
    #=================
    # read the data
    #=================
    fn = 'AuAg_filt.mat'
    
    fn_abs= os.path.join( os.path.dirname( os.path.abspath( __file__)), fn)
    
    data_mat = scio.loadmat(fn_abs)
    p_h = data_mat['zMaps'].transpose(1, 2, 0) # shape P*M*N, P is the number of angles
    p_e = {'Au': data_mat['eMapsAu'].transpose(1, 2, 0), 'Ag': data_mat['eMapsAg'].transpose(1, 2, 0)} # P is the number of angles
    elements = ('Au', 'Ag')
    
    #=================
    # bin the data if necessary
    #=================
    size_bin = 3
    p_e = {el: image_bin(p_e[el], size_bin) for el in elements}
    p_h = image_bin(p_h, size_bin)
    
    #=================
    # define the geometries
    #================
    size_e = next(iter(p_e.values())).shape[-1]
    size_h = p_h.shape[-1]
    tilt_eds = np.linspace(-75.0/180.0*np.pi, 75.0/180.0*np.pi, 31)
    tilt_hd = tilt_eds
    
    if slc > size_e:
        raise ValueError('The indicated slice number is out of range.')
    
    if dim == '2D':
        proj_geom_e = astra.create_proj_geom('parallel', 1.0, size_e,tilt_eds)
        proj_geom_p = astra.create_proj_geom('parallel', 1.0, size_h,tilt_hd)
        vol_geom = astra.create_vol_geom(*[size_e]*2)
    elif dim == '3D':
        proj_geom = astra.create_proj_geom('parallel3d', 1.0,1.0, size_e, size_e, tilt_eds)
        proj_geom_p = astra.create_proj_geom('parallel3d', 1.0, size_h, 1.0, size_h,tilt_hd)
        vol_geom = astra.create_vol_geom(*[size_e]*3)
    else:
        raise ValueError('The dimension is not correct.')
        
    geom_e = Geometries(vol_geom, proj_geom_e)
    geom_p = Geometries(vol_geom, proj_geom_p)
    
    if show:
        callback = (odl.solvers.CallbackPrintIteration(step = 10) & odl.solvers.CallbackShow(step = 10))
    else:
        callback = None

    #================
    # run the reconstruction function
    #================
    
    rec = {el:[] for el in elements}
    dnp_e = {el:[] for el in elements}

    # define the geometry
    if dim == '3D':
        dnp_hd = p_h
        
        for element in elements:
            dnp_e[element] = p_e[element]
    
    elif dim == '2D':
        
        dnp_hd = p_h[:,:, slc]
        
        for element in elements:
            dnp_e[element] = p_e[element][:,:, slc]
               
    data_hd = Projection_data(dnp_hd, geom_p)
    data_hd.normalize() # Normalize the data before reconstruct
    rec_hd = data_hd.reconstruct( norm = 'L2', regularization = 'TV', alg = 'PDHG',  n_iter = 100, lmbd = 1000.0)

    #==================
    # save the figure
    #==================
    save_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'temp')) 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    scms.imsave(os.path.join(save_dir, 'haadf_tv.png'), rec_hd/rec_hd.max())

    # Now try the DR algorithm
    rec_hd = data_hd.reconstruct( norm = 'KL', regularization = 'TV', alg = 'DR',  n_iter = 100, lmbd = 1000.0)

    #==================
    # save the figure
    #==================
    save_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'temp')) 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    scms.imsave(os.path.join(save_dir, 'haadf_tv_DR.png'), rec_hd/rec_hd.max())


    # for element in elements:
    #     data_eds = projection_data( dnp_e[element], geom_e)          
    #     data_eds.normalize()

    #     # reconstruct for every element
    #     rec[element] = wrapper_dg_TNV(data_eds, data_hd, nit= nit, reg_par = lmbd, callback=callback)
 
