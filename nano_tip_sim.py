#Import Headers
import meep as mp
import numpy as np
import h5py
import sys
from pathlib import Path

#Functions for calculating index of refraction -- meep style
def calc_sig_d(n, k, fcen):
    eps = (n + 1j*k)**2
    eps_r = np.real(eps)
    return 2*np.pi*fcen*np.imag(eps)/eps_r

def calc_eps_r(n, k):
    eps = (n + 1j*k)**2
    return np.real(eps)

def run_simulation(save_prefix,
                   tip_radius=0.007,
                   cone_height=0.364,
                   trunk_radius=0.275,
                   n_tip=3.694,
                   k_tip=0.0,
                   fcen=1.25,
                   waves=7.5,
                   theta_deg=10,
                   sample=20,
                   sx=0.8,
                   sy=1.5,
                   sz=1.0,
                   dpml=0.1,
                   res=1000,
                   res_factor=0.2,
                   X_1=-0.2,
                   X_2=0.2,
                   Y_1=-0.2,
                   Y_2=0.2,
                   Z_1=-0.2,
                   Z_2=0.2):

    #Interpolate to next resolution step for high-res region
    dx = 1/res
    X_1 = np.floor(X_1/dx)*dx
    X_2 = np.ceil(X_2/dx)*dx
    Y_1 = np.floor(Y_1/dx)*dx
    Y_2 = np.ceil(Y_2/dx)*dx
    Z_1 = np.floor(Z_1/dx)*dx
    Z_2 = np.ceil(Z_2/dx)*dx

    #Dump all the settings to a file:
    settings_file = h5py.File(Path(sys.argv[0]).stem +
                              '-' + save_prefix +
                              '_settings.h5', 'w')
    settings_file.create_dataset('tip_radius', data=tip_radius)
    settings_file.create_dataset('cone_height', data=cone_height)
    settings_file.create_dataset('trunk_radius', data=trunk_radius)
    settings_file.create_dataset('n_tip', data=n_tip)
    settings_file.create_dataset('k_tip', data=k_tip)
    settings_file.create_dataset('fcen', data=fcen)
    settings_file.create_dataset('waves', data=waves)
    settings_file.create_dataset('theta_deg', data=theta_deg)
    settings_file.create_dataset('sample', data=sample)
    settings_file.create_dataset('sx', data=sx)
    settings_file.create_dataset('sy', data=sy)
    settings_file.create_dataset('sz', data=sz)
    settings_file.create_dataset('dpml', data=dpml)
    settings_file.create_dataset('res', data=res)
    settings_file.create_dataset('res_factor', data=res_factor)
    settings_file.create_dataset('X_1', data=X_1)
    settings_file.create_dataset('X_2', data=X_2)
    settings_file.create_dataset('Y_1', data=Y_1)
    settings_file.create_dataset('Y_2', data=Y_2)
    settings_file.create_dataset('Z_1', data=Z_1)
    settings_file.create_dataset('Z_2', data=Z_2)
    settings_file.close();
    
    #Convert theta to radians
    theta = theta_deg*np.pi/180

    eps_tip = calc_eps_r(n_tip, k_tip)
    sig_d_tip = calc_sig_d(n_tip, k_tip, fcen)

    #Create the cell size
    sX = 2*dpml + sx;
    sY = 2*dpml + sy;
    sZ = 2*dpml + sz;

    sx_prime = sx*res_factor + (X_2 - X_1)*(1 - res_factor)
    sy_prime = sy*res_factor + (Y_2 - Y_1)*(1 - res_factor)
    sz_prime = sz*res_factor + (Z_2 - Z_1)*(1 - res_factor)
    dpml_prime = dpml*res_factor

    sX_prime = 2*dpml_prime + sx_prime
    sY_prime = 2*dpml_prime + sy_prime
    sZ_prime = 2*dpml_prime + sz_prime

    cell = mp.Vector3(sX_prime, sY_prime, sZ_prime)



    tip = [mp.Cylinder(radius=trunk_radius,
                       center=mp.Vector3(0.0,
                                         -1*sY/4.0 - cone_height/2.0 - tip_radius,
                                         0.0),
                       height=(sY/2.0 - cone_height),
                       axis=mp.Vector3(0.0, 1.0, 0.0)),
           mp.Cone(center=mp.Vector3(0, -1*cone_height/2 - tip_radius, 0),
                   height=cone_height,
                   radius=trunk_radius,
                   radius2=tip_radius,
                   axis=mp.Vector3(0,1,0)),
           mp.Sphere(center=mp.Vector3(0,-1*tip_radius,0),
                     radius=tip_radius)]


    def mat_func(r_prime):

        r_x = r_prime.x
        r_y = r_prime.y
        r_z = r_prime.z

        x_fac = 1
        y_fac = 1
        z_fac = 1

        if(r_prime.x < X_1):
            x_fac = res_factor
            r_x = X_1 + (r_prime.x - X_1)/res_factor
        elif(r_prime.x > X_2):
            x_fac = res_factor
            r_x = X_2 + (r_prime.x - X_2)/res_factor

        if(r_prime.y < Y_1):
            y_fac = res_factor
            r_y = Y_1 + (r_prime.y - Y_1)/res_factor
        elif(r_prime.y > Y_2):
            y_fac = res_factor
            r_y = Y_2 + (r_prime.y - Y_2)/res_factor

        if(r_prime.z < Z_1):
            z_fac = res_factor
            r_z = Z_1 + (r_prime.z - Z_1)/res_factor
        elif(r_prime.z > Z_2):
            z_fac = res_factor
            r_z = Z_2 + (r_prime.z - Z_2)/res_factor

        r = mp.Vector3(r_x, r_y, r_z)
    
        J = np.matrix([[x_fac, 0, 0], [0, y_fac, 0], [0, 0, z_fac]]);

        #Loop through all objects inside of tip and see if point is inside.
        #  if yes -- then set eps_point to tip eps
        #  if no -- then leave it as air
        eps_point = 1.0;
        for kk in range(len(tip)):
            if(mp.is_point_in_object(r_prime, tip[kk])):
                eps_point = eps_tip;
            
    
        eps_transform = eps_point*J*J.transpose()/np.linalg.det(J)
        mu_transform = J*J.transpose()/np.linalg.det(J)
    
        eps_diag = eps_transform.diagonal()
        mu_diag = mu_transform.diagonal()
    
        mat = mp.Medium(epsilon_diag = mp.Vector3(eps_diag[0,0],
                                                  eps_diag[0,1],
                                                  eps_diag[0,2]),
                        mu_diag=mp.Vector3(mu_diag[0,0],
                                           mu_diag[0,1],
                                           mu_diag[0,2]),
                        D_conductivity=sig_d_tip)


        return mat


    #Create source amplitude function:
    ky = fcen*np.sin(theta)
    ky_prime = ky*sY/sY_prime
    def my_amp_func_y(r_prime):

        r_y = r_prime.y
        y_fac = 1/res_factor
        
        if((r_prime.x >= X_1)and(r_prime.x <= X_2)):
            x_fac = 1.0/res_factor
        else:
            x_fac = 1.0

        if(r_prime.y < Y_1):
            y_fac = 1.0
            r_y = Y_1 + (r_prime.y - Y_1)/res_factor
        elif(r_prime.y > Y_2):
            y_fac = 1.0
            r_y = Y_2 + (r_prime.y - Y_2)/res_factor

        if((r_prime.z >= Z_1)and(r_prime.z <= Z_2)):
            z_fac = 1.0/res_factor
        else:
            z_fac = 1.0
        
        J = np.matrix([[x_fac, 0, 0], [0, y_fac, 0], [0, 0, z_fac]]);

        transform = J/np.linalg.det(J);
    
        phase_factor = np.exp(-1*2*1j*np.pi*ky*r_y)
        amp_factor = transform.diagonal()[0, 1]
    
        return amp_factor*phase_factor

    def my_amp_func_z(r_prime):

        r_y = r_prime.y
        y_fac = 1.0/res_factor

        if((r_prime.x >= X_1)and(r_prime.x <= X_2)):
            x_fac = 1.0/res_factor
        else:
            x_fac = 1.0

        if(r_prime.y < Y_1):
            y_fac = 1.0
            r_y = Y_1 + (r_prime.y - Y_1)/res_factor
        elif(r_prime.y > Y_2):
            y_fac = 1.0
            r_y = Y_2 + (r_prime.y - Y_2)/res_factor
        

        if((r_prime.z >= Z_1)and(r_prime.z <= Z_2)):
            z_fac = 1.0/res_factor
        else:
            z_fac = 1.0
        
        J = np.matrix([[x_fac, 0, 0], [0, y_fac, 0], [0, 0, z_fac]]);

        transform = J/np.linalg.det(J);
    
        phase_factor = np.exp(-1*2*1j*np.pi*ky*r_y)
        amp_factor = transform.diagonal()[0, 2]
    
        return amp_factor*phase_factor




    #Create PMLs
    pml_layers = [mp.PML(thickness=dpml_prime, direction=mp.X),
                  mp.PML(thickness=dpml_prime, direction=mp.Y),
                  mp.PML(thickness=dpml_prime, direction=mp.Z)]

    symmetry = [mp.Mirror(direction=mp.X)]

    #Sources
    Ey_source = mp.Source(mp.ContinuousSource(frequency=fcen),
                          component=mp.Ey,
                          center=mp.Vector3(0, 0, -1*sz_prime*0.5),
                          size=mp.Vector3(sX_prime, sY_prime, 0),
                          amp_func=my_amp_func_y,
                          amplitude=np.cos(theta))

    Ez_source = mp.Source(mp.ContinuousSource(frequency=fcen),
                          component=mp.Ez,
                          center=mp.Vector3(0, 0, -1*sz_prime*0.5),
                          size=mp.Vector3(sX_prime, sY_prime, 0),
                          amp_func=my_amp_func_z,
                          amplitude=np.sin(theta))

    sources=[Ey_source, Ez_source]

    monitor_xy = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx_prime, sy_prime, 0)) 
    monitor_yz = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(0, sy_prime, sz_prime)) 


    #Now make the simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=[],
                        sources=sources,
                        resolution=res,
                        symmetries=symmetry,
                        dimensions=3,
                        k_point=mp.Vector3(0, -1*ky_prime, 0),
                        material_function=mat_func,
                        extra_materials=[mp.Medium(epsilon=eps_tip,
                                                   mu=4,
                                                   D_conductivity=1)],
                        verbose=True)


    sim.run(mp.in_volume(monitor_xy, mp.to_appended(save_prefix + "E_xy", mp.at_every(1/fcen/sample, mp.output_efield))),
            mp.in_volume(monitor_yz, mp.to_appended(save_prefix + "E_yz", mp.at_every(1/fcen/sample, mp.output_efield))), until=waves/fcen)

    
def visualize_fields_yz(fields_file, settings_file):

    settings_data = h5py.File(settings_file, 'r')
    fields_data = h5py.File(fields_file, 'r')

    tip_radius = np.array(settings_data['tip_radius'])
    X_1 = np.array(settings_data['X_1'])
    X_2 = np.array(settings_data['X_2'])
    Y_1 = np.array(settings_data['Y_1'])
    Y_2 = np.array(settings_data['Y_2'])
    Z_1 = np.array(settings_data['Z_1'])
    Z_2 = np.array(settings_data['Z_2'])

    res = np.array(settings_data['res'])
    res_factor = np.array(settings_data['res_factor'])

    ey = np.array(fields_data['ey.r'])

    #Spacing in real space
    dr = 1/res

    #Get the output size (Y, Z, time)
    cell_size = ey.shape
    z_center = cell_size[1]/2 - 1
    y_center = cell_size[0]/2 - 1

    Z = np.zeros(cell_size[1])
    Y = np.zeros(cell_size[0])
    Ey_fac = np.zeros(cell_size)

    for k in range(0, cell_size[1]):

        Z[k] = (k - z_center)*dr

        if(Z[k] > Z_2):
            Z[k] = (Z[k] - Z_2)/res_factor + Z_2
        elif(Z[k] < Z_1):
            Z[k] = (Z[k] - Z_1)/res_factor + Z_1
        

    for k in range(0, cell_size[0]):

        Y[k] = (k - y_center)*dr
        Y_fac = 1.0/res_factor

        if(Y[k] > Y_2):
            Y[k] = (Y[k] - Y_2)/res_factor + Y_2
            Y_fac = 1.0
        elif(Y[k] < Y_1):
            Y[k] = (Y[k] - Y_1)/res_factor + Y_1
            Y_fac = 1.0

        Ey_fac[k, :, :] = Y_fac


    ey_out = ey*Ey_fac

    return ey_out, Y, Z
    

def visualize_fields_xy(fields_file, settings_file):

    settings_data = h5py.File(settings_file, 'r')
    fields_data = h5py.File(fields_file, 'r')

    tip_radius = np.array(settings_data['tip_radius'])
    X_1 = np.array(settings_data['X_1'])
    X_2 = np.array(settings_data['X_2'])
    Y_1 = np.array(settings_data['Y_1'])
    Y_2 = np.array(settings_data['Y_2'])
    Z_1 = np.array(settings_data['Z_1'])
    Z_2 = np.array(settings_data['Z_2'])

    res = np.array(settings_data['res'])
    res_factor = np.array(settings_data['res_factor'])

    ey = np.array(fields_data['ey.r'])

    #Spacing in real space
    dr = 1/res

    #Get the output size (X, Y, time)
    cell_size = ey.shape
    x_center = cell_size[0]/2 - 1
    y_center = cell_size[1]/2 - 1

    X = np.zeros(cell_size[0])
    Y = np.zeros(cell_size[1])
    Ey_fac = np.zeros(cell_size)


    for k in range(0, cell_size[0]):

        X[k] = (k - x_center)*dr

        if(X[k] > X_2):
            X[k] = (X[k] - X_2)/res_factor + X_2
        elif(X[k] < X_1):
            X[k] = (X[k] - X_1)/res_factor + X_1
        

    for k in range(0, cell_size[1]):

        Y[k] = (k - y_center)*dr
        Y_fac = 1.0/res_factor

        if(Y[k] > Y_2):
            Y[k] = (Y[k] - Y_2)/res_factor + Y_2
            Y_fac = 1.0
        elif(Y[k] < Y_1):
            Y[k] = (Y[k] - Y_1)/res_factor + Y_1
            Y_fac = 1.0

        Ey_fac[:, k, :] = Y_fac
    

    ey_out = ey*Ey_fac

    return ey_out, X, Y

