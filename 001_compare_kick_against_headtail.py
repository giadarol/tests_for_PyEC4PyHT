import sys, os
BIN=os.path.expanduser('../')
sys.path.append(BIN)


import numpy as np
import pylab as pl

#~ filename = 'headtail_for_test/test_protons/SPS_Q20_proton_check_prb.dat'
filename = 'headtail_for_test/test_protons/SPS_Q20_proton_check_20150212_prb.dat'

n_part_per_turn = 5000

appo = np.loadtxt(filename)

parid = np.reshape(appo[:,0], (-1, n_part_per_turn))
x = np.reshape(appo[:,1], (-1, n_part_per_turn))
xp = np.reshape(appo[:,2], (-1, n_part_per_turn))
y = np.reshape(appo[:,3], (-1, n_part_per_turn))
yp =np.reshape(appo[:,4], (-1, n_part_per_turn))
z = np.reshape(appo[:,5], (-1, n_part_per_turn))
zp = np.reshape(appo[:,6], (-1, n_part_per_turn))
N_turns = len(x[:,0])

pl.close('all')
#~ pl.plot(xp[:, 100], '.-')
#~ pl.show()

# define machine for PyHEADTAIL
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from SPS_custom import SPS
machine = SPS(n_segments = 1, machine_configuration = 'Q20-injection', Q_x=20., Q_y=20.)


# compute sigma x and y
epsn_x = 2.5e-6
epsn_y = 2.5e-6

sigma_x = np.sqrt(machine.beta_x[0]*epsn_x/machine.betagamma)
sigma_y = np.sqrt(machine.beta_y[0]*epsn_y/machine.betagamma)

# define apertures and Dh_sc to simulate headtail conditions
x_aper  = 20*sigma_x
y_aper  = 20*sigma_y
Dh_sc = 2*x_aper/128/2

# ecloud
import PyECLOUD.PyEC4PyHT as PyEC4PyHT
from PyHEADTAIL.particles.slicing import UniformBinSlicer
slicer = UniformBinSlicer(n_slices = 64, n_sigma_z = 3.)

init_unif_edens_flag=1
init_unif_edens=2e11
N_MP_ele_init = 100000
N_mp_max = N_MP_ele_init*4.

nel_mp_ref_0 = init_unif_edens*4*x_aper*y_aper/N_MP_ele_init


ecloud = PyEC4PyHT.Ecloud(L_ecloud=machine.circumference, slicer=slicer, 
				Dt_ref=25e-12, pyecl_input_folder='./drift_sim',
				x_aper=x_aper, y_aper=y_aper, Dh_sc=Dh_sc,
				init_unif_edens_flag=init_unif_edens_flag,
				init_unif_edens=init_unif_edens, 
				N_MP_ele_init=N_MP_ele_init,
				N_mp_max=N_mp_max,
				nel_mp_ref_0=nel_mp_ref_0)
machine.install_after_each_transverse_segment(ecloud)

ecloud.save_ele_distributions_last_track = True
ecloud.save_ele_potential_and_field = True
				
# generate a bunch 
bunch = machine.generate_6D_Gaussian_bunch(n_macroparticles=300000, intensity=1.15e11, epsn_x=epsn_x, epsn_y=epsn_y, sigma_z=0.2)


# replace first particles with HEADTAIL ones
bunch.x[:n_part_per_turn] = x[0,:]
bunch.xp[:n_part_per_turn] = xp[0,:]
bunch.y[:n_part_per_turn] = y[0,:]
bunch.yp[:n_part_per_turn] = yp[0,:]
bunch.z[:n_part_per_turn] = z[0,:]
bunch.dp[:n_part_per_turn] =zp[0,:]

# save id and momenta before track
id_before = bunch.id[bunch.id<=n_part_per_turn]
xp_before = bunch.xp[bunch.id<=n_part_per_turn]
yp_before = bunch.yp[bunch.id<=n_part_per_turn]

rms_err_x_list = []
for ii in xrange(N_turns-1):
	# track
	machine.track(bunch, verbose = True)

	# id and momenta after track
	id_after = bunch.id[bunch.id<=n_part_per_turn]
	xp_after = bunch.xp[bunch.id<=n_part_per_turn]
	z_after = bunch.z[bunch.id<=n_part_per_turn]
	yp_after = bunch.yp[bunch.id<=n_part_per_turn]

	# sort id and momenta after track
	indsort = np.argsort(id_after)
	id_after = np.take(id_after, indsort)
	xp_after = np.take(xp_after, indsort)
	yp_after = np.take(yp_after, indsort)
	z_after = np.take(z_after, indsort)

	pl.figure(100+ii)
	sp1 = pl.subplot(2,1,1)
	pl.plot(z_after,  (100*np.abs((xp_after-xp_before)-(xp[ii+1, :]-xp[0, :]))/np.std(bunch.xp)), '.r')
	pl.grid('on')
	pl.subplot(2,1,2, sharex=sp1)
	pl.plot(z_after,  (xp[ii+1, :]-xp[0, :]), '.r')
	pl.plot(z_after,  (xp_after-xp_before), '.b')
	pl.grid('on')
	rms_err_x = np.std(100*np.abs((xp_after-xp_before)-(xp[ii+1, :]-xp[0, :]))/np.std(bunch.xp))
	pl.suptitle('rms_err = %e'%rms_err_x)
	
	rms_err_x_list.append(rms_err_x)

	pl.show()
	
pl.figure(1000)
pl.plot(rms_err_x_list, '.-')
pl.show()

slices = bunch.get_slices(ecloud.slicer)
show_movie = False
if show_movie:
    import time
    n_frames = ecloud.rho_ele_last_track.shape[0]
    pl.close(1)
    pl.figure(1, figsize=(8,8))
    vmax = np.max(ecloud.rho_ele_last_track[:])
    vmin = np.min(ecloud.rho_ele_last_track[:])
    for ii in xrange(n_frames-1, 0, -1):
        pl.subplot2grid((10,1),(0,0), rowspan=3)
        pl.plot(slices.z_centers, np.float_(slices.n_macroparticles_per_slice)/np.max(slices.n_macroparticles_per_slice))
        pl.xlabel('z [m]');pl.ylabel('Long. profile')
        pl.axvline(x = slices.z_centers[ii], color='r')
        pl.subplot2grid((10,1),(4,0), rowspan=6)
        pl.pcolormesh(ecloud.spacech_ele.xg*1e3, ecloud.spacech_ele.yg*1e3, ecloud.rho_ele_last_track[ii,:,:].T, vmax=vmax, vmin=vmin)
        pl.xlabel('x [mm]');pl.ylabel('y [mm]')
        pl.axis('equal')
        pl.subplots_adjust(hspace=.5)
        pl.draw()
        time.sleep(.2)
        pl.show()


