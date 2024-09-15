from multiprocessing import Pool
import numpy as np
import h5py as h5
import random as rd

h5f = 'baikal_muatm-multi_0523_flat_h0_s0.h5'
h5f = '/home3/ivkhar/Baikal/data/filtered/'+h5f

particles = ['muatm','nuatm','nue2']
num_files_to_use = {'muatm':20,'nuatm':10,'nue2':5}

set_Q_hi_lim = True
Q_hi_lim = 100

norm_name = '' # some id

n_workers = 10

def get_means(args):
    (h5_in, part, particle) = args
    with h5.File(h5_in,'r') as hf:
        data = hf[particle+'/data/'+part+'/data'][()]
        if set_Q_hi_lim:
            data[:,0] = np.mimimum( data[:,0], Q_hi_lim )
        mean = np.mean(data, axis=0, dtype=np.float64)
    return (part, mean)

def get_stds(args):
    (h5_in, part, particle, gl_mean) = args
    with h5.File(h5_in,'r') as hf:
        data = hf[particle+'/data/'+part+'/data'][()]
        if set_Q_hi_lim:
            data[:,0] = np.mimimum( data[:,0], Q_hi_lim )
        std = np.sqrt( np.mean(((data-gl_mean))**2, axis=0, dtype=np.float64) )
    return (part, std)

def main():
    with h5.File(h5f,'a') as hf:
        # get numbers of events
        nums_dict = {}
        parts_dict = {}
        for particle in particles:
            nums_dict[particle] = {}
            parts = list(hf[particle+'/data/'].keys())
            parts_dict[particle] = rd.sample(parts, k=num_files_to_use[particle])   
            for part in parts_dict[particle]:
                nums_dict[particle][part] = hf[particle+'/data/'+part+'/data'].shape[0]
        # get local avg
        means = []
        nums = []
        for particle in particles:
            ## gather data quickly
            p_args = [ (h5f, part, particle) for part in parts_dict[particle] ]
            with Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(get_means, p_args):
                    means.append(res[1])
                    nums.append(nums_dict[particle][res[0]])               
        # get global avg
        # calculating factor as this give better precision
        total = sum(nums)
        gl_mean = 0.
        for m,n in zip(means,nums):
            factor = n/total
            gl_mean += m*factor
        hf.create_dataset('norm_param_global/mean'+norm_name, data=gl_mean)
        # get stds
        stds = []
        nums = []
        for particle in particles:
            ## gather data quickly
            p_args = [ (h5f, part, particle, gl_mean) for part in parts_dict[particle] ]
            with Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(get_stds, p_args):
                    stds.append(res[1])
                    nums.append(nums_dict[particle][res[0]])
        # get global std
        # a lot of events and big stds, hence intoducing factor should be good
        gl_std = 0.
        for s,n in zip(stds,nums):
            factor = n/total
            gl_std += np.power(s,2)*factor
        gl_std = np.sqrt(gl_std)
        hf.create_dataset('norm_param_global/std'+norm_name, data=gl_std)

if __name__ == "__main__":
   main()