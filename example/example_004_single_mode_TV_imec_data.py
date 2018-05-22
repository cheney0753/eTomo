from etomo.objects import (Projection_data, Geometries)
from etomo.utils import (image_bin,)
import argparse
import os
import scipy.misc as scms
import scipy.io as scio
import scipy.ndimage as scim
import numpy as np
import astra
import odl

def smooth_gaussian(proj_images, sigma = 0.8):
    
    for i, img in enumerate(proj_images):
        proj_images[i, :, :] = scim.filters.gaussian_filter(img, sigma)
    
    return proj_images

if __name__ == '__main__':
    # difine the parser
    
    pars = argparse.ArgumentParser()
    
    pars.add_argument("--nit", type=int, default = 100, help = "The number of iterations.")
    pars.add_argument('-s', "--slice", type=int, default = -1, help = "The number of reconstructed slice. When the number is -1, the entire volume is reconstructed. Otherwise the slice indicated by the number is reconstructed.")
    pars.add_argument("--dim", type=int, default = 2, help = "The reconstruction dimension: 3D or 2D (slice by slice). 2: 2D; 3: 3D. Default is 2D.")
    pars.add_argument("--lmbd", type=float, default = 1e-1, help = "The regularization paratmeter.")
    pars.add_argument("--show", default = False, action ='store_true')
    pars.add_argument('-o', "--outputdir", type = str, default = '.', help = 'Output directory, defult is current directory')
    pars.add_argument('-i', "--inputdir", type=str, default = '.', help = 'Input directory')
    
    args = pars.parse_args()
    nit = args.nit
    dim = str(args.dim)+'D'
#    dim = '3D'
    slc = args.slice
    lmbd = args.lmbd
    show = args.show
    save_dir = os.path.abspath(args.outputdir)
    
    #=================
    # read the data
    #=================
    input_dir = os.path.join( os.path.dirname(os.path.abspath(__file__)), 
                             'IMEC_Semiconductor')
    
    fn_em = 'elementalmaps.mat'
    fn_ag = 'angles.mat'
    fn_hd = 'projections.mat'

    emap_mat = scio.loadmat( os.path.join( input_dir, fn_em) )
    proj_mat = scio.loadmat( os.path.join( input_dir, fn_hd) )
    angl_mat = scio.loadmat( os.path.join( input_dir, fn_ag) )

    # p_e is the projection data for EDS 
    # p_h is the projection data for HAADF
    
    p_h = proj_mat['proj_l'] # shape P*M*N, P is the number of angles, p*M is a 2D slice
    p_h -= p_h.min()
    p_e = emap_mat['Ti'] # P is the number of angles
    p_e -= p_e.min()
    angl_e = angl_mat['elementalmaps']
    angl_h = angl_mat['projections']

    elements = ('Ti',)
    print(p_h.shape)
    print(p_e.shape)

    
    
    #=================
    # bin the data if necessary
    #=================
    size_bin = 2
    p_e = image_bin(p_e, size_bin)
    p_h = image_bin(p_h, size_bin)
    
    # smooth the EDS data
    p_e = smooth_gaussian(p_e, sigma = 0.8)
    
    if dim == '2D':
        try:
            assert(slc >= 0)
        except:
            slc = p_h.shape[-1]//2
            print('The slice number is not defined. Set to {}.'.format(slc))

    #=================
    # define the geometries
    #================
    size_e = p_e.shape[-1]
    size_h = p_h.shape[-1]
    tilt_e = angl_e / 180.0 * np.pi
    tilt_h = angl_h / 180.0 * np.pi
    
    if slc > size_e:
        raise ValueError('The indicated slice number is out of range.')
    
    if dim == '2D':
        proj_geom_e = astra.create_proj_geom('parallel', 1.0, size_e,tilt_e)
        proj_geom_h = astra.create_proj_geom('parallel', 1.0, size_h,tilt_h)
        vol_geom = astra.create_vol_geom(*[size_e]*2)
        # = vol_geom = astra.create_vol_geom( size_e, size_e)

    elif dim == '3D':
        proj_geom_e = astra.create_proj_geom('parallel3d', 1.0, 1.0, size_e, size_e, tilt_e)
        proj_geom_h = astra.create_proj_geom('parallel3d', 1.0, 1.0, size_h, size_h,tilt_h)
        vol_geom = astra.create_vol_geom(*[size_e]*3)
    else:
        raise ValueError('The dimension is not correct.')
        
    geom_e = Geometries(vol_geom, proj_geom_e)
    geom_h = Geometries(vol_geom, proj_geom_h)
    
    if show:
        callback = (odl.solvers.CallbackPrintIteration(step = 10) & odl.solvers.CallbackShow(step = 10))
    else:
        callback = None

    #================
    # run the reconstruction function
    #================
    

    # define the data
    if dim == '3D':
        dnp_h = p_h  #dnp is data in numpy array
        for element in elements:
            dnp_e = p_e
    
    elif dim == '2D':
        dnp_h = p_h[:,:, slc]
        for element in elements:
            dnp_e= p_e[:,:, slc]
               
    data_h = Projection_data(dnp_h, geom_h)
#    data_h.normalize() # Normalize the data before reconstruct
    data_h.normalize_01() # Normalize the data before reconstruct
    
    data_e = Projection_data(dnp_e, geom_e)
    data_e.normalize_01() # Normalize the data before reconstruct

#    # DO NOT use
#    rec_h = data_h.reconstruct(norm = 'L2', regularization = 'TV', alg = 'PDHG',  n_iter = 100, lmbd = 1000.0)

#    #==================
#    # save the figure
#    #==================
#    if not os.path.exists(save_dir):
#        os.mkdir(save_dir)
#    if dim is '2D':
#        scms.imsave(os.path.join(save_dir, 'haadf_tv.png'), rec_h/rec_h.max())
#    elif dim is '3D':
#        scms.imsave(os.path.join(save_dir, 'haadf_tv.png'),
#                    rec_h[rec_h.shape[0]//2,:,:]/rec_h.max())
 
    # Now try the DR algorithm
    rec_h_dr = data_h.reconstruct( norm = 'KL', regularization = 'TV', alg = 'DR',  n_iter = 100, lmbd = 1000.0)
    
    #==================
    # save the figure
    #==================
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if dim == '2D':
        scms.imsave(os.path.join(save_dir, 'haadf_tv_DR.png'), rec_h_dr/rec_h_dr.max())
    elif dim == '3D':
        scms.imsave(os.path.join(save_dir, 'haadf_tv_DR.png'), rec_h_dr[rec_h_dr.shape[0]//2,:,:]/rec_h_dr.max())
 
     # Now try the DR algorithm for eds
    rec_e = data_e.reconstruct( norm = 'KL', regularization = 'TV', alg = 'DR',  n_iter = 400, lmbd = 0.010)
    
    #==================
    # save the figure
    #==================
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if dim == '2D':
        scms.imsave(os.path.join(save_dir, 'Ti_tv_DR.png'), rec_e/rec_e.max())
        print('Saved to ', os.path.join(save_dir, 'Ti_tv_DR.png') )
        
    elif dim == '3D':
        scms.imsave(os.path.join(save_dir, 'Ti_tv_DR.png'), rec_e[rec_e.shape[0]//2,:,:]/rec_e.max())
        print('Saved to ', os.path.join(save_dir, 'Ti_tv_DR.png') )