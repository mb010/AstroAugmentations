import albumentations as A
import numpy as np

def create_vla_psf(
    save=False, hours=1, t_int=1, frequency=1.4, 
    pixel_resolution=1.8, configuration="B", size=150):
    """Generates and saves a psf for the VLA.
    Args:
        name (str): 
            Output file path to save psf to.
        hours (float>0): 
            Length of 'observation' (time synthesis).
        t_int (float): 
            Time integration length (in hrs).
        frequency (float): 
            Frequency of the observation
        pixel_resolution (float): 
            Pixel width of the saved psf in (arcsec). 
            Default 1.8 arcsec to match FIRST cutouts.
        configuration (str):
            VLA configuration to use.
    """
    
    ### Reading in antennae positions ###
    RawData = []
    with open('./VLA_raw_antenna_positions.txt') as f:
        for line in f: #Create Array of all data.
            LineArray = line.split()
            RawData.append(LineArray)
    # Split dataset By orientation (West, East and North)
    WAntennae = RawData[1:25]
    EAntennae = RawData[25:49]
    NAntennae = RawData[49:]

    # Split location data into Numpy Arrays of various configurations of the satalites
    ArrayConfiguration = 'B'
    W = np.array([])
    for i in WAntennae:
        if ArrayConfiguration in i:
            W = np.append(W,i[-4:])
    #Shape each matrix, so that each row of data is for one receiver with data columns of Lx(ns), Ly(ns), Lz(ns) and R(m).
    W = np.reshape(W,(len(W)//4,4)).astype('float64') 
    E = np.array([])
    for i in EAntennae:
        if ArrayConfiguration in i:
            E = np.append(E,i[-4:])
    E = np.reshape(E,(len(E)//4,4)).astype('float64')
    N = np.array([])
    for i in NAntennae:
        if ArrayConfiguration in i:
            N = np.append(N,i[-4:])
    N = np.reshape(N,(len(N)//4,4)).astype('float64')
    c = 299792458 #[m/s]
    
    NDist = N[:,:3]*10**(-9)*c #[m]
    EDist = E[:,:3]*10**(-9)*c #[m]
    WDist = W[:,:3]*10**(-9)*c #[m]
    
    N_m = NDist[:,:3]
    E_m = EDist[:,:3]
    W_m = WDist[:,:3]
    antennae = np.concatenate((N_m,E_m))
    antennae = np.concatenate((antennae,W_m))
    
    ### Synthesise UV Coverage ###
    # Place coordinates into boxes to show which are sampled in a mgrid of my choosing. Then FT to save a kernel.
    observation_intervals = np.arange(0, hours, t_int)
    UV_coords = []
    for i in range(antennae.shape[0]):
        for j in range(antennae.shape[0]):
            for h in observation_intervals:
                if i!=j:
                    u, v = single_baseline(
                        antennae[i], antennae[j], HA=hours/2-h, 
                        d_deg=34.0784, frequency=frequency)
                    UV_coords.append([u, v])
    UV = np.stack(UV_coords)
    
    ### Grid UV Coverage ###
    lims = [UV.min(), UV.max()]
    uv_grid = np.mgrid[
        lims[0]:lims[1]:(lims[1]-lims[0])//(size-1), 
        lims[0]:lims[1]:(lims[1]-lims[0])//(size-1)
    ]
    u_resolution = (lims[1]-lims[0])//(size-1)
    v_resolution = (lims[1]-lims[0])//(size-1)

    k_list = np.asarray([
        np.where(
            (uv_grid[0]>u) & (uv_grid[0]<=u+u_resolution) & 
            (uv_grid[1]>v) & (uv_grid[1]<=v+v_resolution),
            1, 0
        ) for u, v in UV])
    weighted_uv_sampling = k_list.sum(axis=0)
    psf = np.fft.fftshift(np.fft.fft2(weighted_uv_sampling))
    
    # Save generated psf
    if type(save) is str:
        np.save(save, psf)
    else:
        return psf

def single_baseline(antenna1, antenna2, frequency=1.4, HA=0, uv=True,d_deg=45):
    """Calculates the UV position of a single pair of antennae"""
    c = 299792458 #units: [m/s]
    frequency = frequency*10**9
    baseline = antenna1-antenna2

    if uv:
        H_rad = 2*np.pi * HA/24 #units: [rad]
        d = 2*np.pi * d_deg/360 #units: [rad]
        baseline_u = (np.sin(H_rad)*baseline[0] + np.cos(H_rad)*baseline[1])*frequency/c
        baseline_v = (
            -np.sin(d)*np.cos(H_rad)*baseline[0] + 
            np.sin(d)*np.sin(H_rad)*baseline[1] + 
            np.cos(d)*baseline[2]
        )*frequency/c
    else:
        baseline_u , baseline_v = baseline[0] , baseline[1]

    return baseline_u, baseline_v #units: [lambda]

if __name__ == "__main__":
    output_kernel = "./kernels/VLA_kernel"
    create_vla_psf(
        save = output_kernel,
        hours = 1,
        t_int = 1,
        configuration = 'B'
    )
    print(f"> Generated default VLA PSF / kernel. Saved to:\n{output_kernel}.npy")