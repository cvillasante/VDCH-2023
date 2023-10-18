'''
This Python script reads in position and force data for a given experiment and then calculates the enthalpic stiffnesses (d_force/d_extension above 30 pN).
This only calculates the inverse slope from the first segment of any cycle--that is, before any unfolding has occurred, and only for the extension phase.

Camila Villasante
10-18-2023
'''

# Import modules
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pickle
import numpy as np
import os

from ForceRampHandler_20231018 import ForceRampHandler

# Create lists to store values in
V507DDimer_3 = []
V507DDimer_20 = []
V507DDimer_0 = []
Dimer_3 = []
Dimer_20 = []
Dimer_0 = []
Anchors = []

# Get list of files to calculate stiffnesses from
rootdir = PATH TO FILES

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Import data
        print('Importing',os.path.join(subdir, file))
        data_df = pd.read_hdf(os.path.join(subdir, file))

        # Parse out desired columns
        ywave = data_df['ywavecut']
        forcewave = data_df['forcewavecut']

        # Find number of cycles
        maxima, _ = find_peaks(forcewave, height=3e-11, distance=2.5e5)

        # Print number of cycles
        n_cyc = len(maxima)
        print('There are',n_cyc,'cycles in this dataset.')

        # Split the series of force-extension cycles into individual cycles
        # The ForceRampHandler class takes care of this, as long as the cycles in the 'cut' time series are all of equal length.
        # We will smooth the traces with a savgol filter of length 101 and order 3, which decreases the bandwidth to about 1 ms.
        ramps = ForceRampHandler(forcewave, ywave, n_cyc, smooth = True, auto = False, savgol_coeff = [101,3])

        # Minimum force cut-off for enthalpic region. Here we used 30 pN.
        cut_min_force = 30 # pN
        
        # Minimum force cut-off for cycle processing.
        min_force = 2 # pN

        # Create list to put the slopes in
        stiffness_list = []

        # Now let's loop through the cycles
        for cycle in range(0,n_cyc):
            pull,release = ramps.returnCycle(cycle,min_force)

            # Identify segment 1
            # Zero the pull and releases. This makes segmentation faster
            pull_ext_zeroed = pull.values - np.min(pull.values)
            pull_force_zeroed = pull['force'].values - min_force

            # Segment pulls by the conformational changes and also return the force at which the conformational changes occur
            segments_pulls,ruptureForce_pulls = ramps.returnSegmentsOneCycle(pull_force_zeroed,pull_ext_zeroed, pull=True, numsdevs=4, force_threshold=3, window=1000)
            
            # Un-zero segments
            for seg in segments_pulls:
                seg.values = seg.values + np.min(pull.values)
                seg['force'].values = seg['force'].values+min_force

            # Now cut force and extension values below 30 pN 
            mask = segments_pulls[0]['force']>cut_min_force
            segments_pulls[0] = segments_pulls[0][mask]

            # If the first segment has any values above 30 pN, proceed
            if len(segments_pulls[0]) > 0:

                # Create dataframe from the pull force and end-to-end distance values
                d_pull = {'force':(segments_pulls[0]['force'].values),'extension':(segments_pulls[0].values)}
                df_pull = pd.DataFrame(d_pull)

                # Round force values to 2 decimal places
                df_pull['force_rounded'] = df_pull['force'].round(decimals=2)

                # For all extensions that correspond to a set of forces rounded together, average them to get one extension value
                df_mean_pull = df_pull.groupby('force_rounded')['extension'].apply(lambda x: x.mean()).rename('mean_extension').reset_index()

                # Apply a Savitzky-Golay filter to the force and extension data to smooth
                if len(df_mean_pull['force_rounded']) > 485: # segment length of 485 corresponds to 5 pN 
                    
                    # Now make the window length equal to the length of the segment
                    if (len(df_mean_pull['force_rounded']) % 2) == 0: # window length cannot be an even number though, so if it is make it odd
                        window = len(df_mean_pull['force_rounded'])-1
                    else:
                        window=len(df_mean_pull['force_rounded'])
                    if window <= 1:
                        pass
                    else:
                        # Now smooth the data using a first order Savitzky-Golay filter with a window size equal to the length of the segment
                        df_mean_pull['force_savgol'] = savgol_filter(df_mean_pull['force_rounded'],window,1)
                        df_mean_pull['extension_savgol'] = savgol_filter(df_mean_pull['mean_extension'],window,1)

                        # Now calculate d_force, d_extension, and stiffness
                        d_force= df_mean_pull['force_savgol'].iloc[-1]-df_mean_pull['force_savgol'].iloc[0]
                        d_extension= df_mean_pull['extension_savgol'].iloc[-1]-df_mean_pull['extension_savgol'].iloc[0]
                        stiffness=d_force/d_extension
                        stiffness_list.append(stiffness)

        # Assign stiffnesses to correct list depending on construct/condition
        if 'V507DDimer-3mM' in file:
            V507DDimer_3.append(stiffness_list)
        elif 'V507DDimer-20uM' in file:
            V507DDimer_20.append(stiffness_list)
        elif 'V507DDimer-0M' in file:
            V507DDimer_0.append(stiffness_list)
        elif 'Dimer-3mM' in file:
            Dimer_3.append(stiffness_list)
        elif 'Dimer-20uM' in file:
            Dimer_20.append(stiffness_list)
        elif 'Dimer-0M' in file:
            Dimer_0.append(stiffness_list)
        elif 'anchors' in file:
            Anchors.append(stiffness_list)


# Make dictionary to contain all stiffnesses
allStiffnesses = {'V507DDimer_3':V507DDimer_3,'V507DDimer_20':V507DDimer_20,'V507DDimer_0':V507DDimer_0,'Dimer_3':Dimer_3,'Dimer_20':Dimer_20,'Dimer_0':Dimer_0,'Anchors':Anchors}
# create a output pickle file 
f1 = open(PATH TO OUTPUT FILE,"wb")

# write the Python object (dict) to pickle file
pickle.dump(allStiffnesses,f1)

# close file
f1.close()
