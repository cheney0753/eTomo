  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:41:20 2018

@author: zhong

etomo.recipe module for the etomo package
"""

import odl
import astra
import numpy as np
from etomo.objects import (Projection_data,)
#=================
# the EDS data module
#=================
class BT_recipe(object):
    """bimodal tomography reconstruction algorithmic recipe class"""
    def __init__(self, projdata_e, projdata_h):
        self.projdata_e = projdata_e
        self.projdata_h = projdata_h
        self.__ingredients = []
        
    def prepare(self, module_data, module_reg = None, module_hebt = None):
        try:
            assert module_data in ('KL', 'L2')
        except AssertionError:
            raise Exception('module_data defined fault.')
        assert module_reg in ('None', 'none', None, 'TV', 'TNV', 'HETNV', 'HTNV')
        assert module_hebt in ('None', 'none', None, 'HEBT')
        self.__ingredients = {'data': module_data,
                                 'reg': module_reg,
                                 'hebt': module_hebt} 

    def cook(self, n_iter, alpha = 0.0, lmbd = 0.0, callback = None, rec_in = None):
        if self.__ingredients['hebt'] == 'HEBT':
            if self.__ingredients['reg'] == 'TV':
                x = dg_solver(self.projdata_e, self.projdata_e, n_iter, self.__ingredients['data'], regl = 'TV', lmbd = lmbd, alpha = alpha, callback = callback)
            if self.___ingredients['reg'] == 'TNV':


    @property
    def ingredients():
        return self.__ingredients_
    
#==========
    # define a function to perform the Douglas-Roughford algorithm using the odl backend
#==========
def dg_solver(projdata_e, projdata_h, nit, norm_e = 'L2', norm_d = 'L2', rec_sup = None,  regl = None, lmbd = 0.0, alpha = 1.0, beta = 1.0, callback = None ):

    """ 
    Solve the reconstruction problem with the recipe defined by the parameters

    Args:
    __________
    nit: int
        The number of iterations
    reg_par_em: float
        Weight of the regularization term defined in emTomo.elemental_maps
    reg_par_hp: float
        Weight of the regularization term defined in emTomo.projections

    Notes:
    __________
    The function solves optimization problem in this general form:

    .. math::
       \{x^{e}\}, x^h = (1-\alpha)\sum_e d( W^e x^e, p^e) + d(w^h x^h, p^h) + \alpha d(\sum_e w^h x^e, p^h) + \lambda r({x^e}, x^h),

    where :math:`d` is the KL or L2 data discrepancy function
    By setting :math:'\alpha' 0, we have:

    .. math::
       \{x^{e}\}, x^h = \sum_e d( W^e x^e, p^e) + d(w^h x^h, p^h) + \lambda r({x^e}, x^h),

    which is defined in the TNV-regularized EDS tomography paper:
    TODO:add links

    Reference:
    __________
    check the odl example code: nuclearn_norm_tomography.py
    https://github.com/odlgroup/odl/blob/a11a69a760359b9482a150554ac4cb4f20bc4b90/examples/solvers/nuclear_norm_tomography.py
    """
    if not regl:
        regl = 'None'
    if not regl in ('TNV', 'TV', 'HETNV', 'HTNV', 'none','None'):
        raise ValueError("The regl must be one of TNV, TV, HTNV, HETNV or None.")
    if regl is 'HTNV' and not rec_sup:
        raise ValueError('rec_sup is not defined for HTNV.')
    if regl is 'HTNV' and beta != 0:
        raise ValueError('beta must be set to 0 for HTNV')
    # if the projdata_e is not a list or tuple, put it in a list
    if not isinstance(projdata_e, (tuple, list)):
        projdata_e = [projdata_e]

    oplist = [e.geometries.ray_trafo for e in projdata_e]
    if beta != 0:
        oplist.append( projdata_h.geometries.ray_trafo)
    diagop = odl.DiagonalOperator(*oplist)

    # domain is a list of the domains of reconstructions:
    # there are n+1 reconstructions, n reconstructions for n elements,
    # and 1 reconstruction for HAADF 
    domain = odl.ProductSpace(*[op.domain for op in oplist])

    # use a reduction operator to incoporate the HEBT term
    redop = odl.ReductionOperator(*[ ri*projdata_h.geometries.ray_trafo for ri in r])

    # Create data discrepancy for every element (x_e's)
    g_e = []
    for e in projdata_e:
        if norm_e == 'KL':
            g_e.append( (1-alpha)*odl.solvers.KullbackLeibler(e.ray_trafo.range, prior=e.data_odl))
        elif norm_e == 'L2':
            g_e.append( (1-alpha)*odl.solvers.L2NormSquared(e.ray_trafo.range).translated(e.data_odl))

    if beta != 0:
        # Create data discrepancy for the projdata_h for x_h
        if norm_d == 'KL':
            g_h = beta*odl.solvers.KullbackLeibler(projdata_h.ray_trafo.range, prior= projdata_h.data_odl)
        elif norm_d == 'L2':
            g_h = beta*odl.solvers.L2NormSquared(projdata_h.ray_trafo.range).translated(projdata_h.data_odl)

    if alpha != 0:
    # Create data discrepancy for the HEBT term
        if norm_d == 'KL':
            g_h = alpha*odl.solvers.KullbackLeibler(projdata_h.ray_trafo.range, prior= projdata_h.data_odl)
        elif norm_d == 'L2':
            g_h = alpha*odl.solvers.L2NormSquared(projdata_h.ray_trafo.range).translated(projdata_h.data_odl)

    # Create regularization functional
    # Gradient
    grad_e = [odl.Gradient(e.ray_trafo.domain) for e in projdata_e]
    grad_h = odl.Gradient(projdata_h.ray_trafo.domain)

    # Set up the nuclear norm.
    if regl in ('HETNV', 'HTNV'):
        diaggr = odl.DiagonalOperator(*grad_e, grad_h)
        g_reg = lmbd*odl.solvers.NuclearNorm(diaggr.range, singular_vector_exp=1)
    elif regl is  'TNV':
        diaggr = odl.DiagonalOperator(*grad_e)
        g_reg = lmbd*odl.solvers.NuclearNorm(diaggr.range, singular_vector_exp=1)
    elif regl is  'TV':
        g_reg = odl.solvers.SeparableSum( *[lmbd*odl.solvers.GroupL1Norm(ge.range] for ge in grad_e)
    else:
        g_reg = None

    # Assemble operators
    lin_ops = [diagop, redop, diaggr]

    # Assemble functionals depending on the alpha and beta
    if alpha == 0 and beta != 0:
        g_data = odl.solvers.SeparableSum(*g_e, g_h)
    elif alpha != 0 and beta == 0:
        g_data = odl.solvers.SeparableSum(*g_e, g_sum)
    elif alph !=0 and beta !=0:
        g_data = odl.solvers.SeparableSum(*g_e, g_h, g_sum)

    # Assemble functionals g
    if g_reg:
        g = [g_data, g_reg]
    else:
        g = [g_data]
                                          

    # add a cost function term to make the last of x is identical to  rec_sub 
    if regl is 'HTNV':
        diag_zero_ind = odl.ReductionOperator(*[odl.ZeroOperator(e.ray_trafo.domain) for e in projdata_e], odl.IdentityOperator( projdata_h.ray_trafo.domain))
        g_zero = odl.solvers.IndicatorZero(projdata_h.ray_trafo.domain).translated(rec_sup)
        lin_ops.append(diag_zero_ind)
        g.append(g_zero)
    # Define function f
    if nonnegativity:
        # Create box constraint functional ( non-negativity constraint)
        f = odl.solvers.IndicatorBox(domain, lower = 0)
    else:
        f = odl.solvers.IndicatorBox(domain) 

    # Compute step length parameters to satisfy the condition
    # (see douglas_rachford_pd) for more info
    tau = 2.0 / len(lin_ops)
    sigma = [1 / odl.power_method_opnorm(op, rtol=0.1)**2 for op in lin_ops]

    # Solve
    x = domain.one()

    odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau=tau, sigma=sigma, niter=nit, callback = callback)

    return x
