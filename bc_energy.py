#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:12:22 2019

@author: liaojw
"""
import numpy as np
import xmitgcm
import xgcm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import os
import xarray as xr
from matplotlib import colors
import seawater as sw
from scipy.interpolate import griddata

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_SETTINGS"] = "0"
#%%
class Buget(object):
    """This class used to calculate the baroclinic energy
       The input data is a dataset
    """
    def __init__(self,dset):
        self.dset = dset
        self.grid = xgcm.Grid(dset)
        self.rhoc = 1029
        self.grav = 9.81
    
        z0 = xr.zeros_like(self.dset.ETAN)
        hzC = (self.dset.hFacC*self.dset.drF+z0).transpose('time','Z','YC','XC')
        hzC = hzC.load()
        hzC[:,0] = hzC[:,0]+self.dset.ETAN
        
        z0 = xr.zeros_like(self.grid.interp(self.dset.ETAN,'X'))
        hzW = (self.dset.hFacW*self.dset.drF+z0).transpose('time','Z','YC','XG')
        hzW = hzW.load()
        hzW[:,0] = hzW[:,0]+self.grid.interp(self.dset.ETAN,'X')

        z0 = xr.zeros_like(self.grid.interp(self.dset.ETAN,'Y'))        
        hzS = (self.dset.hFacS*self.dset.drF+z0).transpose('time','Z','YG','XC')
        hzS = hzS.load()
        hzS[:,0] = hzS[:,0]+self.grid.interp(self.dset.ETAN,'Y')

        
        self.hzC = hzC
        self.hzW = hzW
        self.hzS = hzS
        
    def cal_bc_velocity(self):
        """
        calculate velocity of baroclinic tide
        """
        ub0 =  self.dset.UVEL - self.dset.UVEL.mean('time')
        ubH =  ub0*self.hzW 
        
        ubt = (ubH.sum('Z')/self.grid.interp(self.dset.Depth+self.dset.ETAN,'X')).rename('ubt')        
        ubt = ubt.where(self.dset.UVEL[:,0,:,:].values!=0)
        ubc = (ub0-ubt).rename('ubc')
        ubc = ubc.where(self.dset.UVEL.values!=0)
      
        vb0 =  self.dset.VVEL-self.dset.VVEL.mean('time')        
        vbH =  vb0*self.hzS
        vbt = (vbH.sum('Z')/self.grid.interp(self.dset.Depth+self.dset.ETAN,'Y')).rename('vbt')
        vbt = vbt.where(self.dset.VVEL[:,0,:,:].values!=0)        
        vbc = (vb0-vbt).rename('vbc')  
        vbc = vbc.where(self.dset.VVEL.values!=0)
                
        sigma = (self.dset.Z-self.dset.ETAN)/(self.dset.Depth+self.dset.ETAN)
        sigma.values[sigma.values<-1]= np.nan
        
        diff_Dx = self.grid.diff(self.dset.Depth+self.dset.ETAN,'X')*\
                  self.dset.dyG/self.dset.rAw
        diff_Dy = self.grid.diff(self.dset.Depth+self.dset.ETAN,'Y')*\
                  self.dset.dxG/self.dset.rAs
        
        diff_etax = self.grid.diff(self.dset.ETAN,'X')*self.dset.dyG/self.dset.rAw
        diff_etay = self.grid.diff(self.dset.ETAN,'Y')*self.dset.dxG/self.dset.rAs
        
        eta = self.dset.ETAN.load()
        diff_etat = (eta.shift(time=-1)-eta)/3600 #!frequency
        diff_etat[-1] = diff_etat[-2]
        
        wbt_u = ubt*(self.grid.interp(sigma,'X')*diff_Dx+diff_etax)
        wbt_v = vbt*(self.grid.interp(sigma,'Y')*diff_Dy+diff_etay)
#        wbt_u = ubt*(np.array(sigma)*diff_Dx+diff_etax)
#        wbt_v = vbt*(np.array(sigma)*diff_Dy+diff_etay)

        wbt = self.grid.interp(wbt_u,'X')+self.grid.interp(wbt_v,'Y')+(sigma+1)*diff_etat
        wbt = wbt.rename('wbt')
        wbt = wbt.transpose('time','Z','YC','XC')
        
        w   = self.grid.interp(self.dset.WVEL,'Z')
        wbc = (w-wbt).rename('wbc')
        return xr.merge([ubc,vbc,ubt,vbt,wbt,wbc])
    def cal_rhobc(self):
        nt,nz,ny,nx = self.dset.THETA.shape
        
        tesz1 = (np.tile(self.dset.Z,(ny,1)))
        tesz2 = (np.tile(tesz1,(nx,1,1))).T
        tesz3 = np.tile(tesz2,(nt,1,1,1))
        
        pdens = sw.dens(self.dset.SALT,self.dset.THETA,abs(tesz3))

        rho = xr.DataArray(pdens,dims = self.dset.THETA.dims,\
                                 coords = self.dset.THETA.coords)
        rhobc = rho-rho.mean('time')
        return rhobc
    def cal_rhobc_model(self):
        rho = self.dset.RHOAnoma+self.rhoc
        rhotm = rho.mean('time')
        rhobc = rho-rhotm
        return rhobc
    def cal_pbc(self):
        rhobc = self.cal_rhobc()
        pbc0 = self.grav*rhobc*self.hzC
        pbc0 = pbc0.transpose('time','Z','YC','XC')
        nt,nz,ny,nx = pbc0.shape
        pbc1 = np.zeros((nt,nz,ny,nx))
        pbc1 = xr.DataArray(pbc1,dims=('time','Z','YC','XC'),\
                           coords={'time':pbc0.time,'Z':pbc0.Z,'YC':pbc0.YC,\
                                    'XC':pbc0.XC})
        for i in np.arange(1,nz+1):
            pbc1[:,i-1] = pbc0[:,:i].sum('Z')
        psurf = (pbc1*self.hzC).sum('Z')/(self.dset.Depth+self.dset.ETAN)
        pbc = pbc1 - psurf
        return pbc
    def cal_pbc_model(self):
        presure = (self.dset.PHIHYD+abs(self.grav*self.dset.Z))*self.rhoc
        ptm = presure.mean('time')
        p0 = presure - ptm  
        pH = (p0*self.hzC).sum('Z')
        pbc = p0-pH/(self.dset.Depth+self.dset.ETAN)
        
        return pbc
    
    def cal_bfrq2(self):
        nt,nz,ny,nx = self.dset.THETA.shape
        tesz1 = (np.tile(self.dset.Z,(ny,1)))
        tesz2 = (np.tile(tesz1,(nx,1,1))).T
        bfrq2 = np.zeros((nt,nz,ny,nx))
        for i in np.arange(nt):
            bfrq21 = sw.bfrq(self.dset.SALT[i],self.dset.THETA[i],abs(tesz2))
            bfrq22 = bfrq21[0]
            bfrq22[bfrq22<=1e-8] = np.nan
            bfrq2[i,1:] = bfrq22
            bfrq2[i,0] = bfrq2[i,1]
        bfrq2[self.dset.THETA.values==0]=np.nan
        bfrq2_xr = xr.DataArray(bfrq2,dims = self.dset.THETA.dims,\
                                      coords = self.dset.THETA.coords)
        return bfrq2_xr
    
    def cal_all_terms(self):
        rhobc = self.cal_rhobc()
        bfrq2 = self.cal_bfrq2()
        velbc = self.cal_bc_velocity()
        pbc = self.cal_pbc()

        
#   calculate APE
        APE = 0.5*self.grav**2*rhobc**2/bfrq2/self.rhoc
        APE = APE.where(bfrq2.values!=np.nan)        
        APE = APE.rename('APE')
        APE_tmzs = ((APE*self.hzC).sum('Z').mean('time')).rename('APE_tmzs')
#   calculate KE    
        KE = 0.5*self.rhoc*(self.grid.interp(velbc.ubc,'X')**2+\
                            self.grid.interp(velbc.vbc,'Y')**2+\
                            velbc.wbc**2) 
        KE = KE.where(bfrq2.values!=np.nan)        
        KE = KE.rename('KE')
        KE_tmzs = ((KE*self.hzC).sum('Z').mean('time')).rename('KE_tmzs')
#   calculate KE0
        
        KE0 = self.rhoc*self.grid.interp(velbc.ubt*velbc.ubc,'X')+\
              self.rhoc*self.grid.interp(velbc.vbt*velbc.vbc,'Y')
              
            
#    calculate dAPE/dt,dKE/dt
        APE_tm = (APE*self.hzC).sum('Z')
        dAPE = (APE_tm[-1]-APE_tm[0])/(len(self.dset.time)*3600)
        KE_tm  = (KE*self.hzC).sum('Z')
        dKE  = (KE_tm[-1]-KE_tm[0])/(len(self.dset.time)*3600)
        Tend_tmzs = (dAPE+dKE).rename('Tend_tmzs')
#    calculate Ebt2bc
        Ebt2bc = self.grav*(rhobc*velbc.wbt*self.hzC).sum('Z')
        Ebt2bc_tmzs = (Ebt2bc.mean('time')).rename('Ebt2bc_tmzs')
       
        fx_APE = (velbc.ubt*self.grid.interp(APE,'X')*(self.hzW)).sum('Z')
        fy_APE = (velbc.vbt*self.grid.interp(APE,'Y')*(self.hzS)).sum('Z')
                
        fx_KE = (velbc.ubt*self.grid.interp(KE,'X')*(self.hzW)).sum('Z')
        fy_KE = (velbc.vbt*self.grid.interp(KE,'Y')*(self.hzS)).sum('Z')        
        
        fx_KE0 = (velbc.ubt*self.grid.interp(KE0,'X')*(self.hzW)).sum('Z')
        fy_KE0 = (velbc.vbt*self.grid.interp(KE0,'Y')*(self.hzS)).sum('Z')        
        
#        fx_adv_tmzs = ((fx_KE).mean('time')).rename('fx_adv_tmzs')
#        fy_adv_tmzs = ((fy_KE).mean('time')).rename('fy_adv_tmzs')
        
        fx_adv_tmzs = ((fx_APE+fx_KE+fx_KE0).mean('time')).rename('fx_adv_tmzs')
        fy_adv_tmzs = ((fy_APE+fy_KE+fy_KE0).mean('time')).rename('fy_adv_tmzs')
        
        div_adv_x = self.grid.diff(fx_adv_tmzs*self.dset.dyG,'X')/self.dset.rA
        div_adv_y = self.grid.diff(fy_adv_tmzs*self.dset.dxG,'Y')/self.dset.rA
        
        divflux_adv = (div_adv_x+div_adv_y).rename('divflux_adv')

# calculate pressure work of flux        
        pbc_u = self.grid.interp(pbc,'X')
        pbc_v = self.grid.interp(pbc,'Y')
        
#        flux_x = (velbc.ubc*pbc_u*(self.dset.drF*self.dset.hFacW+\
#                                   self.grid.interp(self.dset.ETAN,'X'))).sum('Z')
#        flux_y = (velbc.vbc*pbc_v*(self.dset.drF*self.dset.hFacS+\
#                                    self.grid.interp(self.dset.ETAN,'Y'))).sum('Z')
        flux_x = (velbc.ubc*pbc_u*self.hzW).sum('Z')
        flux_y = (velbc.vbc*pbc_v*self.hzS).sum('Z')
        
        fx_pre_tmzs = (flux_x.mean('time')).rename('fx_pre_tmzs')
        fy_pre_tmzs = (flux_y.mean('time')).rename('fy_pre_tmzs')

        div_pre_x = self.grid.diff(fx_pre_tmzs*self.dset.dyG,'X')/self.dset.rA
        div_pre_y = self.grid.diff(fy_pre_tmzs*self.dset.dxG,'Y')/self.dset.rA

        divflux_pre = (div_pre_x+div_pre_y).rename('divflux_pre')

# calculate diffusion of flux
        vish = 0.01
        visz = 1e-5
        kh = 0.01
        kz = 1e-5
        Cd = 0.0025

        fx_dif = ((vish*self.grid.diff(KE,'X')*self.dset.dyG/self.dset.rAw+\
                       kh*self.grid.diff(APE,'X')*self.dset.dyG/self.dset.rAw)*(self.hzW)).sum('Z')
        fy_dif = ((vish*self.grid.diff(KE,'Y')*self.dset.dxG/self.dset.rAs+\
                       kh*self.grid.diff(APE,'Y')*self.dset.dxG/self.dset.rAs)*(self.hzS)).sum('Z')        
        
        fx_dif_tmzs = (fx_dif.mean('time')).rename('fx_dif_tmzs')
        fy_dif_tmzs = (fy_dif.mean('time')).rename('fy_dif_tmzs')
        
        div_dif_x = self.grid.diff(fx_dif_tmzs*self.dset.dyG,'X')/self.dset.rA
        div_dif_y = self.grid.diff(fy_dif_tmzs*self.dset.dxG,'Y')/self.dset.rA
        
        divflux_dif = (-(div_dif_x+div_dif_y)).rename('divflux_dif')

        fx_tmzs = (fx_adv_tmzs+fx_pre_tmzs-fx_dif_tmzs).rename('fx_tmzs')
        fy_tmzs = (fy_adv_tmzs+fy_pre_tmzs-fy_dif_tmzs).rename('fy_tmzs')
        divflux_tmzs = (divflux_adv + divflux_pre+divflux_dif).rename('divflux_tmzs')

#%%calculate disbuget with baroclinic budget
        Dis_tmzs_budget = (Ebt2bc_tmzs - divflux_tmzs - Tend_tmzs).rename('Dis_tmzs_budget')
#%%calculate dissipation directly
        ubc_x = self.grid.diff(velbc.ubc*self.dset.dyG,'X')/self.dset.rA
        ubc_y = self.grid.diff(self.grid.interp(self.grid.interp(velbc.ubc,'Y'),'X')*self.dset.dxG,'Y')/self.dset.rA
        ubc_z = self.grid.interp((self.grid.interp(self.grid.diff(velbc.ubc,'Z'),'Z')/self.dset.drF),'X')

        vbc_x = self.grid.diff(self.grid.interp(self.grid.interp(velbc.vbc,'X'),'Y')*self.dset.dyG,'X')/self.dset.rA
        vbc_y = self.grid.diff(velbc.vbc*self.dset.dxG,'Y')/self.dset.rA
        vbc_z = self.grid.interp((self.grid.interp(self.grid.diff(velbc.vbc,'Z'),'Z')/self.dset.drF),'Y')
        
        w_x   = self.grid.diff(self.dset.WVEL,'X')/self.dset.dxC
        w_x   = self.grid.interp(w_x,'X')
        w_x   = self.grid.interp(w_x,'Z')
        
        w_y   = self.grid.diff(self.dset.WVEL,'Y')/self.dset.dyC
        w_y   = self.grid.interp(w_y,'Y')
        w_y   = self.grid.interp(w_y,'Z')
        
        w_z   = self.grid.diff(self.dset.WVEL,'Z')/self.dset.drF
        
        dish  = self.rhoc*vish*(ubc_x**2+ubc_y**2+vbc_x**2+vbc_y**2)+\
                self.rhoc*visz*(ubc_z**2+vbc_z**2)
        disv  = self.rhoc*vish*(w_x**2+w_y**2)+self.rhoc*visz*w_z**2
        
        uu    = self.grid.interp(self.dset.UVEL*velbc.ubc,'X')
        vv    = self.grid.interp(self.dset.VVEL*velbc.vbc,'Y')
        ww    = self.grid.interp(self.dset.WVEL,'Z')**2
        uh    = self.grid.interp(velbc.ubt,'X')
        vh    = self.grid.interp(velbc.vbt,'Y')
        D     = self.rhoc*Cd*abs(uh)*(uu+vv+ww)+\
                self.rhoc*Cd*abs(vh)*(uu+vv+ww)
        ny,nx = self.dset.Depth.shape
        D = D.transpose('time','Z','YC','XC')
        D.values[self.dset.UVEL.values==0]=0
        D0 = np.array(D)
        D1 = np.zeros_like(D0)

        for i in np.arange(ny):
            for j in np.arange(nx):
                if np.isnan(D0[0,0,i,j]):
                    continue
                else:
                    lo = np.where(np.isnan(D0[0,:,i,j]))           
                    D1[:,lo[0][0]-1,i,j] = D0[:,lo[0][0]-1,i,j]
        D2 = xr.zeros_like(D)
        D2.values = D1
        D2        = D2.rename('D2')
        
        eps   =  dish+disv
        Dis_model = (eps).rename('Dis_model')
        
        Dis_total_model = (eps+D2).rename('Dis_total_model')
        
        eps_h = (eps*self.hzC).sum('Z')
#calculate the dissipation of eddy diffusivity
        rhobc_x = self.grid.diff(rhobc,'X')/self.dset.dxC
        rhobc_y = self.grid.diff(rhobc,'Y')/self.dset.dyC
        rhobc_z = self.grid.interp(self.grid.diff(rhobc,'Z'),'Z')/self.dset.drF
        
        rhobc_xx = self.grid.interp(rhobc_x**2,'X')
        rhobc_yy = self.grid.interp(rhobc_y**2,'Y')
        rhobc_zz = rhobc_z**2
        Eddy_dif = self.grav**2/(self.rhoc*bfrq2)*kh*(rhobc_xx+rhobc_yy)+\
                   self.grav**2/(self.rhoc*bfrq2)*kz*rhobc_zz
        
        Eddy_dif_h = (Eddy_dif*self.hzC).sum('Z')
           
        Dis_tmzs_model = (eps_h.mean('time')).rename('Dis_tmzs_model')
        Dis_tmzs_total_model = ((eps_h+D2.sum('Z')+Eddy_dif_h).mean('time')).rename('Dis_tmzs_total_model')
        
        return xr.merge([APE,KE,APE_tmzs,KE_tmzs,Tend_tmzs,Ebt2bc_tmzs,\
                         divflux_tmzs,divflux_adv,divflux_pre,divflux_dif,\
                         fx_tmzs,fy_tmzs,fx_adv_tmzs,fy_adv_tmzs,\
                         fx_pre_tmzs,fy_pre_tmzs,fx_dif_tmzs,fy_dif_tmzs,\
                         Dis_tmzs_budget,Dis_model,Dis_tmzs_model,\
                         Dis_total_model,Dis_tmzs_total_model,D2])
	
#%%
if __name__ == '__main__':

    print('Calculate the budget of baroclinic tide')
    print('test 24 degree')

    dir_exp = '/public/home/liaojw/Andaman/MITgcm_65y/verification/internal_tides/'

#    exp = 'test.48_degree'
    exp = 'run2.sum'

    day = [16,17]
#    day=[18,25]
    frequency=3600;deltaTmom=90;
    deltastep=frequency//deltaTmom;
    iters=list(np.arange((day[0]-1)*24*deltastep+deltastep/2,(day[1]-1)*24*deltastep+deltastep/2,deltastep))

    ds_24d = xmitgcm.open_mdsdataset(dir_exp+exp,iters=iters,prefix=['Surf','Stat','Vel','Pres'],delta_t=90)
    grid=xgcm.Grid(ds_24d)
    
    result = Buget(ds_24d)
    terms = result.cal_all_terms()
#%%
    Tend_tmzs = terms.Tend_tmzs
    Ebt2bc = terms.Ebt2bc_tmzs
    divflux_tmzs = terms.divflux_tmzs
    divflux_adv = terms.divflux_adv
    divflux_pre = terms.divflux_pre
    divflux_dif = terms.divflux_dif
    fx_tmzs,fy_tmzs = terms.fx_tmzs,terms.fy_tmzs
    APE_tmzs = terms.APE_tmzs
    KE_tmzs  = terms.KE_tmzs
    Dis_tmzs_budget = terms.Dis_tmzs_budget
    Dis_tmzs_model  = terms.Dis_tmzs_model
    Dis_tmzs_total_model = terms.Dis_tmzs_total_model
#%%

    
    fx_adv_tmzs,fy_adv_tmzs = terms.fx_adv_tmzs,terms.fy_adv_tmzs
    fx_pre_tmzs,fy_pre_tmzs = terms.fx_pre_tmzs,terms.fy_pre_tmzs
    fx_dif_tmzs,fy_dif_tmzs = terms.fx_dif_tmzs,terms.fy_dif_tmzs
#%%
    dnm = '7_1'    
    output_dir = '/public/home/liaojw/Andaman/MITgcm_65y/verification/internal_tides/output/'+exp+'/14day/'+dnm

    xmitgcm.utils.write_to_binary(Tend_tmzs.values.flatten(),output_dir+'/Tend_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(Ebt2bc.values.flatten(),output_dir+'/Ebt2bc_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(divflux_tmzs.values.flatten(),output_dir+'/Divflux_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(divflux_adv.values.flatten(),output_dir+'/Divflux_adv_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(divflux_pre.values.flatten(),output_dir+'/Divflux_pre_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(divflux_dif.values.flatten(),output_dir+'/Divflux_dif_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(fx_tmzs.values.flatten(),output_dir+'/fx_tmzs_'+dnm+'.bin')    
    xmitgcm.utils.write_to_binary(fy_tmzs.values.flatten(),output_dir+'/fy_tmzs_'+dnm+'.bin')  
    xmitgcm.utils.write_to_binary(fx_adv_tmzs.values.flatten(),output_dir+'/fx_adv_tmzs_'+dnm+'.bin')    
    xmitgcm.utils.write_to_binary(fy_adv_tmzs.values.flatten(),output_dir+'/fy_adv_tmzs_'+dnm+'.bin')   
    xmitgcm.utils.write_to_binary(fx_pre_tmzs.values.flatten(),output_dir+'/fx_pre_tmzs_'+dnm+'.bin')    
    xmitgcm.utils.write_to_binary(fy_pre_tmzs.values.flatten(),output_dir+'/fy_pre_tmzs_'+dnm+'.bin')   
    xmitgcm.utils.write_to_binary(fx_dif_tmzs.values.flatten(),output_dir+'/fx_dif_tmzs_'+dnm+'.bin')    
    xmitgcm.utils.write_to_binary(fy_dif_tmzs.values.flatten(),output_dir+'/fy_dif_tmzs_'+dnm+'.bin')   

    xmitgcm.utils.write_to_binary(Dis_tmzs_model.values.flatten(),output_dir+'/Dis_tmzs_model_'+dnm+'.bin')
    xmitgcm.utils.write_to_binary(Dis_tmzs_total_model.values.flatten(),output_dir+'/Dis_tmzs_total_model_'+dnm+'.bin')
    


