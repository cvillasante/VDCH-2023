'''
This python script reads in position and force data for a given experiment, segments each cycle,
and then fits all segments with our model. Then the difference in x_E parameters between subsequent segments is calculated.  

Camila Villasante
10-17-2023
'''
# Import modules
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import time
import pickle
import numpy as np
from scipy.optimize import curve_fit

from TimeSeries_20230111 import ForceRampHandler
from TimeSeries_20230111 import TimeSeriesLoader as tsl

import os
import glob

# These stiffness values are for construct AND toy peptide in series, not the construct alone, because we are observing the extension of both in our experiments
k_tot_V507DDimer_3 = 3.186901577468575 # mN/m
k_tot_V507DDimer_20 = 2.1119903354348923 # mN/m
k_tot_V507DDimer_0 = 1.5920404706803661 # mN/m
k_tot_Dimer_3 = 3.7387282651447826 # mN/m
k_tot_Dimer_20 = 2.4711717398138338 # mN/m
k_tot_Dimer_0 = 2.387027314211473 # mN/m

# Upper bound of f_half as determined by adding the IQR*1.5 to the third quartile of all f_HALF values from all segments and all datasets while only constraining K 
f_half_upper_bound = 1/1.8716400552236159 #pN

def Saturation2paramFK(F,x_E,f_half):
    K = K_to_use
    ext = x_E/(1+(f_half/(F))) + (F/K)

    return ext

def Saturation2paramStateX(F,x_E):
    f_half = f_half_s[0]
    K = K_to_use 

    ext = x_E/(1+(f_half/(F))) + (F/K)

    return ext

def Saturation2paramStateXReleases(F,x_E):
    f_half = f_half_s_releases[0]
    K = K_to_use

    ext = x_E/(1+(f_half/(F))) + (F/K)

    return ext

# Cycle to start at
start = 0

# Minimum force cut-off (we used 2 pN, the data are noisy below that)
min_force = 2 #pN

# Get list of files to do stuff with
rootdir = PATH TO FILES

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))

        # Let's select the correct stiffness value to use
        if 'V507DDimer-3mM' in file:
            print('Lets use V507D Dimer 3 mM Ca2+ K value!')
            K_to_use = k_tot_V507DDimer_3
        elif 'V507DDimer-20uM' in file:
            print('Lets use V507D Dimer 20 uM Ca2+ K value!')
            K_to_use = k_tot_V507DDimer_20
        elif 'V507DDimer-0M' in file:
            print('Lets use V507D Dimer No Ca2+ K value!')
            K_to_use = k_tot_V507DDimer_0
        elif 'Dimer-3mM' in file:
            print('Lets use Dimer 3 mM Ca2+ K value!')
            K_to_use = k_tot_Dimer_3
        elif 'Dimer-20uM' in file:
            print('Lets use Dimer 20 uM Ca2+ K value!')
            K_to_use = k_tot_Dimer_20
        elif 'Dimer-0M' in file:
            print('Lets use Dimer No Ca2+ K value!')
            K_to_use = k_tot_Dimer_0

        # Import data
        data_df = pd.read_hdf(os.path.join(subdir, file))

        # Parse out desired columns
        ywave = data_df['ywavecut']
        forcewave = data_df['forcewavecut']

        # Find number of cycles in dataset
        maxima, _ = find_peaks(forcewave, height=3e-11, distance=2.5e5)

        # Print number of cycles
        n_cyc = len(maxima)
        print('There are',n_cyc,'cycles in this dataset.')  

        # Create ramps object
        ramps = ForceRampHandler(forcewave, ywave, n_cyc, smooth = True, auto = False, savgol_coeff = [101,3])

        # Create lists that we will write things to
        dx_Es_all_pulls = []
        dx_Es_all_releases = []

        ruptureForcePullsList = []
        ruptureForceReleasesList = []

        dx_Es_vs_ruptureForce_pulls = []
        dx_Es_vs_ruptureForce_releases = []

        sum_dx_Es_all_pulls = []
        sum_dx_Es_all_releases = []

        sum_dx_Es_all_pulls_and_events = []
        sum_dx_Es_all_releases_and_events = []

        dx_Es_all_pulls_vs_cycle = []
        dx_Es_all_releases_vs_cycle = [] 
        
        # Now let's loop through the cycles in the dataset
        for cycle in range(start,n_cyc):
            print('This is cycle',cycle)

            # Get cycle
            pull,release = ramps.returnCycle(cycle,min_force)
            
            # Zero the pull and release. This makes the segmentation by conformational changes faster.
            pull_ext_zeroed = pull.values - np.min(pull.values)
            release_ext_zeroed = release.values - np.min(release.values)
            pull_force_zeroed = pull['force'].values - min_force
            release_force_zeroed = release['force'].values - min_force

            # Segment pulls by the conformational changes and also return the force at which the conformational changes occur
            segments_pulls,ruptureForce_pulls = ramps.returnSegmentsOneCycle(pull_force_zeroed,pull_ext_zeroed, pull=True, numsdevs=4, force_threshold=3, window=1000)
            print('There are',len(segments_pulls),'pull segments in this cycle.')
            if len(ruptureForce_pulls) > 1:
                ruptureForce_pulls.pop()
                ruptureForce_pulls = [x + min_force for x in ruptureForce_pulls]
                ruptureForcePullsList.append(ruptureForce_pulls)
                print('The rupture forces of the pull segments are:',ruptureForce_pulls)
            
            # Segment releases by the conformational changes and also return the force at which the conformational changes occur
            segments_releases,ruptureForce_releases = ramps.returnSegmentsOneCycle(release_force_zeroed,release_ext_zeroed, pull=False, numsdevs=4, force_threshold=3, window=1000)
            print('There are',len(segments_releases),'release segments in this cycle.')
            if len(ruptureForce_releases) > 1:
                ruptureForce_releases.pop()
                ruptureForce_releases = [x + min_force for x in ruptureForce_releases]
                ruptureForceReleasesList.append(ruptureForce_releases)
                print('The rupture forces of the release segments are:',ruptureForce_releases)

            # Un-zero the pull and release
            for seg in segments_pulls:
                seg.values = seg.values + np.min(pull.values)
                seg['force'].values = seg['force'].values+min_force
            
            for seg in segments_releases:
                seg.values = seg.values + np.min(release.values)
                seg['force'].values = seg['force'].values+min_force
            
            x_E_s = []
            f_half_s = []
            for i, seg in enumerate(segments_pulls):
                if i == 0:
                    guesses = (seg.values.max()-seg.values.min(),1/0.7)
                    params,cov = curve_fit(Saturation2paramFK,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=((0),(np.inf,f_half_upper_bound)))
                    x_E_s.append(params[0])
                    f_half_s.append(params[1])
                else:
                    guesses = (seg.values.max()-seg.values.min())
                    params,cov = curve_fit(Saturation2paramStateX,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=(0,np.inf))
                    x_E_s.append(params[0])
            if len(x_E_s) > 1: # if there are unfolding events...
                dx_E_s = np.diff(x_E_s)
                dx_Es_all_pulls.append(dx_E_s)
                dx_Es_vs_ruptureForce_pulls.append(list(zip(ruptureForce_pulls,dx_E_s)))
                dx_Es_all_pulls_vs_cycle.append(list((dx_E_s,cycle)))
                sum_dx_Es_all_pulls.append(sum(dx_E_s))
                sum_dx_Es_all_pulls_and_events.append((sum(dx_E_s),len(dx_E_s)))
                print('Unfolding events in the extension phase of cycle',cycle,':',dx_E_s)
                print('Total unfolding in cycle',cycle,'is:',sum(dx_E_s),'nm.')
                print('Rupture forces and their unfolding events:',list(zip(ruptureForce_pulls,dx_E_s)))
            x_E_s_releases = []
            f_half_s_releases = []
            if len(segments_releases[0]) > len(segments_releases[-1]): # this makes sure that the longest segment of release will be the one used for fitting. Segment 0 is the one closest to the max force
                for i, seg in enumerate(segments_releases):
                    if i == 0:
                        guesses = (seg.values.max()-seg.values.min(),1/0.7)
                        params,cov = curve_fit(Saturation2paramFK,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=((0,0),(np.inf,f_half_upper_bound)))
                        x_E_s_releases.append(params[0])
                        f_half_s_releases.append(params[1])
                    else:
                        guesses = (seg.values.max()-seg.values.min())
                        params,cov = curve_fit(Saturation2paramStateXReleases,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=(0,np.inf))
                        x_E_s_releases.append(params[0])
            else:
                for i, seg in enumerate(reversed(segments_releases)): # reverse the segments, because segment 0 is the "baseline" segment
                    if i == 0:
                        guesses = (seg.values.max()-seg.values.min(),1/0.7)
                        params,cov = curve_fit(Saturation2paramFK,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=((0,0),(np.inf,f_half_upper_bound)))
                        x_E_s_releases.append(params[0])
                        f_half_s_releases.append(params[1])
                    else:
                        guesses = (seg.values.max()-seg.values.min())
                        params,cov = curve_fit(Saturation2paramStateXReleases,seg['force'],seg.values,p0=guesses,absolute_sigma=True,maxfev=10000,bounds=(0,np.inf))
                        x_E_s_releases.append(params[0])
            if len(x_E_s_releases) > 1: # if there are unfolding events...
                x_E_s_releases.reverse() # flip so that the subtraction goes the right way!
                dx_E_s_releases = np.diff(x_E_s_releases)
                dx_Es_all_releases.append(dx_E_s_releases)
                dx_Es_vs_ruptureForce_releases.append(list(zip(ruptureForce_releases,dx_E_s_releases)))
                dx_Es_all_releases_vs_cycle.append(list((dx_E_s_releases,cycle)))
                sum_dx_Es_all_releases.append(sum(dx_E_s_releases))
                sum_dx_Es_all_releases_and_events.append((sum(dx_E_s_releases),len(dx_E_s_releases)))
                print('Unfolding events in the relaxation phase of cycle',cycle,':',dx_E_s_releases)
                print('Total unfolding in cycle',cycle,'is:',sum(dx_E_s_releases),'nm.')
                print('Rupture forces and their unfolding events:',list(zip(ruptureForce_releases,dx_E_s_releases)))


        output_dict = {'dx_Es_all_pulls':dx_Es_all_pulls,'dx_Es_all_releases':dx_Es_all_releases,
                    'ruptureForcePullsList':ruptureForcePullsList,'ruptureForceReleasesList':ruptureForceReleasesList,
                    'dx_Es_vs_ruptureForce_pulls':dx_Es_vs_ruptureForce_pulls,'dx_Es_vs_ruptureForce_releases':dx_Es_vs_ruptureForce_releases,
                    'sum_dx_Es_all_pulls':sum_dx_Es_all_pulls,'sum_dx_Es_all_releases':sum_dx_Es_all_releases,
                    'sum_dx_Es_all_pulls_and_events':sum_dx_Es_all_pulls_and_events,'sum_dx_Es_all_releases_and_events':sum_dx_Es_all_releases_and_events,
                    'dx_Es_all_pulls_vs_cycle':dx_Es_all_pulls_vs_cycle,'dx_Es_all_releases_vs_cycle':dx_Es_all_releases_vs_cycle}
        
        outfile = open(os.path.join(PATH TO DIRECTORY,file),"wb")

        # Write the python object (dict) to a pickle file
        pickle.dump(output_dict,outfile)

        # Close file
        outfile.close()

    