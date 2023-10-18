# ForceRampHandler class to deal with optical trap data
# Originally written by Dr. Tobias Bartsch during 2015-2019
# Modified by Camila Villasante and Xinyue Deng during 2020-2023

# Import modules
import numpy as np
import xarray as xr
from scipy.signal import find_peaks as find_peaks
from scipy.signal import savgol_filter

class ForceRampHandler(object):
    '''Provides tools to handle single-molecule force ramp data.'''

    def __init__(self, forcewave, extensionwave, numtrials, smooth = True, auto = True, savgol_coeff = [101,3]):
        '''Create a new ForceRampHandler.

        Args:
            forcewave (xarray.Dataarray): an appropriately cut force time trace (must contain only ramps, nothing else).
            extensionwave (xarray.Dataarray): an appropriately cut extension time trace (over the same time domain as the forcewave).
            numtrials (int): number of ramps in the time trace
            smooth (Boolean): smooth the data using a Savitzky Golay filter?
            savgol_coeff (list): [savgol window size, polynomial order]
        '''
        #smooth?
        if(smooth):
            extensionwave = savgol_filter(extensionwave, savgol_coeff[0],savgol_coeff[1])
            forcewave = savgol_filter(forcewave, savgol_coeff[0],savgol_coeff[1]) 
         
        if(auto == False):
            #stack each individual phase of each cycle into its own row of a matrix
            length = int(np.floor(len(np.array(forcewave))/(numtrials*2)))
            fwave = forcewave[0:numtrials*2*length]
            exwave = extensionwave[0:numtrials*2*length]
            forcematrix = np.array(fwave).reshape((-1, length))
            extensionmatrix = np.array(exwave).reshape((-1, length))
            force_pulls = forcematrix[::2]
            force_releases = forcematrix[1::2]
            extension_pulls = extensionmatrix[::2]
            extension_releases = extensionmatrix[1::2]
        if(auto == True):
            smoothed_signal = forcewave[::100]
            new = np.convolve(smoothed_signal, np.ones((50,))/50, mode='valid')
            diff = np.diff(new)
            new_diff = np.convolve(diff, np.ones((50,))/50, mode='valid')
            new_diff_2 = abs(np.diff(new_diff))
            max_idx = find_peaks(new_diff_2,height=max(new_diff_2)/5,distance=300)[0]*100
            max_idx = np.reshape(max_idx,(int(len(max_idx)/3),3))
            size = int((max_idx[0][2] - max_idx[0][0] + 10000)/2)
            force_pulls,force_releases,extension_pulls,extension_releases = [np.zeros(size)],[np.zeros(size)],[np.zeros(size)],[np.zeros(size)]
            for val in max_idx:
                val[1] = val[1] + 5000
                force_pull = np.array(forcewave[val[1] - size:val[1]])
                force_release = np.array(forcewave[val[1]:val[1] + size])
                ex_pull = np.array(extensionwave[val[1] - size:val[1]])
                ex_release = np.array(extensionwave[val[1]:val[1] + size])
                force_pulls = np.append(force_pulls,[force_pull],axis=0)
                force_releases = np.append(force_releases,[force_release],axis=0)
                extension_pulls = np.append(extension_pulls,[ex_pull],axis=0)
                extension_releases = np.append(extension_releases,[ex_release],axis=0)
            force_pulls = force_pulls[1:]
            force_releases = force_releases[1:]
            extension_pulls = extension_pulls[1:]
            extension_releases = extension_releases[1:]
        
        # Switch variables so that force is independent variable and extension is the dependent variable
        self.pulls = np.stack((force_pulls, extension_pulls), axis=2)
        self.releases = np.stack((force_releases, extension_releases), axis=2)


    @property
    def pullsXr(self):
        '''Return a list of xarray.DataArray objects for all pulls'''
        pulls = []
        for pull in self.pulls:
            pullArray = xr.DataArray(pull[:,1],
                                    dims=['force'],
                                    coords = {'force': pull[:,0]},
                                    name='extension')
            pulls.append(pullArray)
        return pulls

    @property
    def releasesXr(self):
        '''Return a list of xarray.DataArray objects for all releases'''
        releases = []
        for release in self.releases:
            releaseArray = xr.DataArray(release[:,1],
                                    dims=['force'],
                                    coords = {'force': release[:,0]},
                                    name='extension')
            releases.append(releaseArray)
        return releases
    
    def returnCycle(self, wanted_cycle, min_force=2):
        '''Return two dataframes, one containing all pulls data (force and extension) and the other containing all releases data
        
        Args:
            start (int): number of first cycle in layout.
            stop (int): number of last cycle in layout.
            mine_force(int): get rid of points with forces below this value

        Returns:
            dataframes: dataframes containing all pulls and releases (columns)
        '''

        pull_cycle = self.pullsXr[wanted_cycle]
        pull_cycle = pull_cycle[pull_cycle['force'].values>min_force]
        release_cycle = self.releasesXr[wanted_cycle]
        release_cycle = release_cycle[release_cycle['force'].values>min_force]

        return pull_cycle, release_cycle
        
    def _detectConfChange(self, forcewave, extensionwave, pull=True, numsdevs=4, force_threshold=3, window=1000): 
        '''Runs a statistical test to determine conformational changes in the extension wave. Only considers data for which the force is larger than force_threshold.

            Args:
                forcewave (np.array): force time trace.
                extensionwave (np.array): extension time trace.
                pull (boolean): Is the provided data a pull or a relaxation?
                numsdevs (int): identify events if they are this many number of sdevs above noise.
                force_threshold (float): threshold for the force: do not test data with a force smaller than this value.
                window (int): number of data points to consider for the statistical test.
            Returns:
                steps_start (np.array): detected start indices of conformational changes.
                steps_end (np.array): detected end indices of conformational changes.
        '''
        if(pull is False): #create waves with monotonically increasing force
            forcewave = np.flip(forcewave)
            extensionwave = np.flip(extensionwave)
        
        mask = forcewave > force_threshold
        detected_steps = np.copy(forcewave) 
        detected_steps.fill(0)
        
        #where does the ramp start?
        startindex = np.argmax(mask)

        for index in np.arange(startindex + window, len(extensionwave) - window, 1):
            mean_before = np.mean(extensionwave[index-window:index])
            mean_after =  np.mean(extensionwave[index:index+window])
            sdev_before = np.std(extensionwave[index-window:index])
            sdev_after = np.std(extensionwave[index:index+window])
            detected_steps[index] = np.abs(mean_after-mean_before) > numsdevs*(sdev_before+sdev_after)/2
        
        #undo the flip:
        if(pull is False):
            detected_steps = np.flip(detected_steps)

        #find beginning and end of each transition
        d_steps = np.diff(detected_steps)
        steps_start = np.where(d_steps==1)
        steps_end = np.where(d_steps==-1)
        return (steps_start, steps_end)

    def _confChangeToSegments(self, steps_start, steps_end, forcewave, extensionwave,pull=True):
        '''Return a segmentated representation fo forcewave and extension wave, deliminated by the conformational changes

            Args:
                steps_start (np.array): detected start indices of conformational changes, output from _detectConfChange
                steps_end (np.array): detected end indices of conformational changes, output from _detectConfChange
                forcewave (np.array): force time trace
                extensionwave (np.array): extension time trace 
            
            Returns:
                segments (list): list of xr.DataArray containing the force extension segments.
        '''
        _steps_start = np.insert(steps_start, 0, 0) #insert a zero at the beginning of _steps_start
        _steps_end = np.insert(steps_end, 0, 0)
        
        _steps_start = np.append(_steps_start, len(forcewave))
        _steps_end = np.append(_steps_end, len(forcewave))

        ruptureForce = []
        segments = []
        for start, end in zip(_steps_start[1:], _steps_end[0:-1]):
            force_segment = forcewave[end:start]
            extension_segment = extensionwave[end:start]
            seg = xr.DataArray(extension_segment,#force_segment,
                                dims=['force'],#dims=['extension'],
                                coords={'force': force_segment},#coords={'extension': extension_segment},
                                name='extension')#name='force')
            if pull==True:
                ruptureForce.append(np.max(seg['force'].values))#ruptureForce.append(np.max(seg.values))
            else:
                ruptureForce.append(np.min(seg['force'].values))#ruptureForce.append(np.min(seg.values))
            segments.append(seg)

        return segments,ruptureForce

    def returnSegmentsOneCycle(self,force_cycle,extension_cycle, pull=True, numsdevs=4, force_threshold=3, window=1000):
        '''
        Return segments (demarcated by conformational change) of a pull/release curve of interest.
        
        Args: 
            force_cycle (pandas dataframe): force wave of the pull or release.
            extension_cycle (pandas dataframe): extension wave of the pull or release
            pull (boolean): Is the provided data a pull or a relaxation?
            numsdevs (int): identify events if they are this many number of sdevs above noise.
            force_threshold (float): threshold for the force: do not test data with a force smaller than this value.
            window (int): number of data points to consider for the statistical test.
        
        Returns:
            segments (list): list of xr.DataArray containing the force extension segments.
        '''

        (steps_start,steps_end) = self._detectConfChange(force_cycle, extension_cycle, pull, numsdevs=numsdevs, force_threshold=force_threshold, window=window)
        (segments, ruptureForce) = self._confChangeToSegments(steps_start, steps_end, force_cycle, extension_cycle,pull)

        return segments, ruptureForce
    
    






    
