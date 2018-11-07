#Import Headers
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

#----------------
# Settings
#----------------

#Save Settings
save_prefix = "t_shift_175mrad_r1000_7nmrad"

#Tip Properties
tip_radius = 0.007
cone_height = 0.364
trunk_radius = 0.275

#Variable resolution properties
X_1 = -1.1*tip_radius
X_2 = 1.1*tip_radius
Y_1 = -2*tip_radius
Y_2 = 2*tip_radius
Z_1 = -1.1*tip_radius
Z_2 = 1.1*tip_radius

sx = 0.8;
sy = 1.5;
sz = 1.0;
dpml = 0.1;

sample = 20;
fcen = 1.25;
res = 1000;
res_factor = 0.2;
waves = 7.5;
theta = 0.175;

#Tip Properties
n_tip = 3.694
#n_tip = 1.0
k_tip = 0.0
#k_tip = 0.0065435


#---------------------
# Calcs
#---------------------

#Functions for calculating index of refraction -- meep style
def calc_sig_d(n, k, fcen):
    eps = (n + 1j*k)**2
    eps_r = np.real(eps)
    return 2*np.pi*fcen*np.imag(eps)/eps_r

def calc_eps_r(n, k):
    eps = (n + 1j*k)**2
    return np.real(eps)

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


def mat_func(r):

    r_prime_x = r.x
    r_prime_y = r.y
    r_prime_z = r.z

    x_fac = 1
    y_fac = 1
    z_fac = 1

    if(r.x < X_1):
        x_fac = res_factor
        r_prime_x = X_1 + (r.x - X_1)/res_factor
    elif(r.x > X_2):
        x_fac = res_factor
        r_prime_x = X_2 + (r.x - X_2)/res_factor

    if(r.y < Y_1):
        y_fac = res_factor
        r_prime_y = Y_1 + (r.y - Y_1)/res_factor
    elif(r.y > Y_2):
        y_fac = res_factor
        r_prime_y = Y_2 + (r.y - Y_2)/res_factor

    if(r.z < Z_1):
        z_fac = res_factor
        r_prime_z = Z_1 + (r.z - Z_1)/res_factor
    elif(r.z > Z_2):
        z_fac = res_factor
        r_prime_z = Z_2 + (r.z - Z_2)/res_factor

    r_prime = mp.Vector3(r_prime_x, r_prime_y, r_prime_z)
        
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
def my_amp_func_y(r):

    r_prime_y = r.y
    y_fac = 1
    
    if((r.x >= X_1)and(r.x <= X_2)):
        x_fac = 1
    else:
        x_fac = res_factor

    if(r.y < Y_1):
        y_fac = res_factor
        r_prime_y = Y_1 + (r.y - Y_1)/res_factor
    elif(r.y > Y_2):
        y_fac = res_factor
        r_prime_y = Y_2 + (r.y - Y_2)/res_factor

    if((r.z >= Z_1)and(r.z <= Z_2)):
        z_fac = 1
    else:
        z_fac = res_factor
        
    J = np.matrix([[x_fac, 0, 0], [0, y_fac, 0], [0, 0, z_fac]]);

    transform = J/np.linalg.det(J);
    
    phase_factor = np.exp(-1*2*1j*np.pi*ky*r_prime_y)
    amp_factor = transform.diagonal()[0, 1]
    
    return amp_factor*phase_factor

def my_amp_func_z(r):

    r_prime_y = r.y
    y_fac = 1

    if((r.x >= X_1)and(r.x <= X_2)):
        x_fac = 1
    else:
        x_fac = res_factor

    if(r.y < Y_1):
        y_fac = res_factor
        r_prime_y = Y_1 + (r.y - Y_1)/res_factor
    elif(r.y > Y_2):
        y_fac = res_factor
        r_prime_y = Y_2 + (r.y - Y_2)/res_factor
        

    if((r.z >= Z_1)and(r.z <= Z_2)):
        z_fac = 1
    else:
        z_fac = res_factor
        
    J = np.matrix([[x_fac, 0, 0], [0, y_fac, 0], [0, 0, z_fac]]);

    transform = J/np.linalg.det(J);
    
    phase_factor = np.exp(-1*2*1j*np.pi*ky*r_prime_y)
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


sim.run(mp.in_volume(monitor_xy, mp.to_appended(save_prefix + "Ey_xy", mp.at_every(1/fcen/sample, mp.output_efield_y))),
        mp.in_volume(monitor_yz, mp.to_appended(save_prefix + "Ey_yz", mp.at_every(1/fcen/sample, mp.output_efield_y))), until=waves/fcen)



