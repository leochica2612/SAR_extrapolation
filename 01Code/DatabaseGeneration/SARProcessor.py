# IMPORTS
import netCDF4 as nc
import os
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter
import glob
import pickle
import numpy as np
from itertools import combinations
from .configManager import Config

from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from .algorithms.cmod5_n import calc_sigma0_cmod5_n
from .algorithms.wind_profile import diabatic_wind_profile, delta_diabatic_wind_profile

# SAR file processor class
"""
    This class extracts useful information from pre-processed L-1 SLC Sentinel-1 AB data.
    
    All SAR files to be processed by this class should be pre-processed with SNAP software according to graph_merge_AS_remote.xml
    They should be in netcdf format.

    It can take as input a single SAR file or a folder where all files are stored.

    The available computations are:
    - Selection of tile of interest within the SAR image
        - Description: Select a tile of interest as set by lat, lon coordinates and width and height.
        - Function: obtain_target_tile
    - Computation of the 2D power spectral density
        - Description: Compute the 2D PSD of the selected tile of interest. It requires the tile selection first.
        - Function : compute_fft_2D, and compute_welch_2D
    - Removal of solid objects
        - Description: Application of a false alarm filter in order to remove targets such as ships or wind turbines from the SAR image.
        - Function: filter_objects
    - Intersection with FINO1 data
        - Description: Find the FINO1 data record corresponding to the SAR image measurement time
        - Function: get_closest_measurement
"""
class SARProcessor:

    def __init__(self, config: Config):
        # sar_src_path,sar_dst_path='',sar_file_name='',lat=0,lon=0,width=3500,height=3500
        """
        Class constructor.

        Params:
        sar_src_path : String
            Path to the folder where all pre-processed SLC products are stored
        sar_dst_path : String (optional)
            Path to the folder where all products processed by this class will be stored if saved by write_pickle function.
        sar_file_name : String (optional)
            Path to a file to be processed if only a single one is to be analyzed.
        lat : float
            Target latitude (deg). All computations will be centered around this latitude.
        lon : float
            Target longitude (deg). All computations will be centered around this longitude.
        width : int
            Target tile width in pixels
        height : int
            Target tile height in pixels
        """

        # Save paths as attributes
        self.src_path = config.sar_src_path #sar_src_path
        self.sar_dst_path = config.sar_dst_path #sar_dst_path

        # Create SAR dst directory
        dirname, fname = os.path.split(self.sar_dst_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # Add target coordinates
        self.lat = config.lat
        self.lon = config.lon

        # Add the width and height of the analysis window
        self.width = config.width
        self.height = config.height

    ############## METHODS ############
    def process_sar_files(self):
        """
            Process the SAR .nc files within src_path. Tiles at the coordinates of interest with dimensions
            width and height are cutted out from the complete image, thus reducing the dataset weight.
            
            IMPORTANT! this function requires that fino_path is defined and pointing to the FINO1 dataframe
        """

        # Get all the available files in src_path
        files = glob.glob(os.path.join(self.src_path,'*.nc'))

        # Process each .nc SAR measurement file
        for file in files:
            print(f'Processing file {file}')
            try:
                # Read the .nc file
                self.read_file(file)

                # Cut the target tile to be processed
                self.obtain_target_tile()

                # Filter out objects in the tile such as ships or wind turbines
                self.filter_objects(num_guard=20, num_ref=20, pfa=1e-3)

                # Compute the 2D PSDs of the tile using the Welch method and periodogram method
                # self.compute_welch_2D(tile_size=(128, 128), overlap=0.5, window='hamming', return_db=True)
                self.compute_fft_2D()

                self.compute_autocorr_2D()

                # Delete the dataset containing the whole SAR image and only retain the cutted tile
                del self.ds

                # Save the processed tile
                self.write_product(os.path.basename(file).split('.')[0])
            except:
                print(f'Couldnt process file {file}')

    def process_single_sar_file(self,filename):
            """
                Process a single SAR .nc file within src_path. Tiles at the coordinates of interest with dimensions
                width and height are cutted out from the complete image, thus reducing the dataset weight.
                
                IMPORTANT! this function requires that fino_path is defined and pointing to the FINO1 dataframe
            """
    
            # Get all the available files in src_path
            file = os.path.join(self.src_path,filename)
    
            # Process each .nc SAR measurement file
            print(f'Processing file {file}')
            # try:
            # Read the .nc file
            self.read_file(file)

            # Cut the target tile to be processed
            self.obtain_target_tile()

            # Filter out objects in the tile such as ships or wind turbines
            self.filter_objects(num_guard=20, num_ref=20, pfa=1e-3)

            # Compute the 2D PSDs of the tile using the Welch method and periodogram method
            # self.compute_welch_2D(tile_size=(128, 128), overlap=0.5, window='hamming', return_db=True)
            self.compute_fft_2D()

            self.compute_autocorr_2D()

            # Delete the dataset containing the whole SAR image and only retain the cutted tile
            del self.ds

            # Save the processed tile
            self.write_product(os.path.basename(file).split('.')[0])
            return self.tile
            # except:
            #     print(f'Couldnt process file {file}')

    def print_info(self):
            """
            Get the variable names and their dimensions from a NetCDF file.
    
            Parameters:
            file_name : str
                Name of the NetCDF file to read.
    
            Returns:
            var_info : dict
                Dictionary containing variable names as keys and their dimensions as values.
            """
            print(f"Getting variable names and dimensions from NetCDF file: {self.file_path}")
            with nc.Dataset(f"{os.path.join(self.src_path, self.file_path)}", 'r') as dataset:
                var_info = {var: (dataset.variables[var].dimensions, dataset.variables[var].shape) for var in dataset.variables}
            print(var_info)

    def write_product(self,filename):
        """
            Function to save processed tiles as pickles
        
        Params:
        filename : String
            File name of the pickle to be saved.
        """

        # Create the pickle path
        path = os.path.join(self.sar_dst_path,filename+'.nc')

        # Add psd to the tile
        self.tile['psd'] = self.psd
        
        # Add autocorr to the tile
        self.tile['autocorr'] = self.autocorr

        # Save the tile pickle
        self.tile.to_netcdf(path)
             
    def read_file(self, file_path=''):
        """
            Function to read a SAR file as an xarray dataset.
        
        Params:
        file_path : String
            path of the .nc SAR file to be read
        Returns:
        self.ds : xr.Dataset
            Dataset containing all the information in the SAR .nc file
        """

        # Check if the input file_path is to be used, or the on in the class attributes
        if os.path.exists(file_path):
            self.ds = xr.open_dataset(file_path)
        else:
            print("File path doesn't exist")
            return

        # Add icident angle and platform heading for easy access
        self.ds['incident_angle'] = self.ds['incident_angle']
        self.ds['platform_heading'] = [
            val for key, val in self.ds.metadata.attrs.items() if key.endswith("platformHeading")
        ]
        self.ds['platform_heading'] = self.ds['platform_heading'][0]
        return self.ds

    def haversine_distance(self, lat1, lon1, lat2, lon2, earth_radius_km=6371.0):
        """
        Calculate the Haversine (great-circle) distance between two points on Earth.

        Parameters:
        -----------
        lat1, lon1 : float or np.ndarray
            Latitude and longitude of the first point(s) in degrees.
        lat2, lon2 : float or np.ndarray
            Latitude and longitude of the second point(s) in degrees.
        earth_radius_km : float, optional
            Radius of the Earth in kilometers (default is 6371.0 km).
            Use 3958.8 for miles.

        Returns:
        --------
        float or np.ndarray
            Great-circle distance in kilometers.
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        # Differences in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arcsin(np.sqrt(a))

        return earth_radius_km * c

    def find_minimum_distance(self, ds, target_lat, target_lon):
        """
        Function to obtain the minimum distance of an input dataset to the target latitude and longitude

        Params:
        ds : xr.Dataset
            The dataframe containing the SAR-measured data to carry out the minimum distance search
        target_lan : float
            Target latitude (deg)
        target_lon : float
            Target longitude (deg)

        Returns:
        x_idx : int
            Index in the x dimension of the ds corresponding to the minimum distance to target coordinates
        y_idx : int
            Index in the y dimension of the ds corresponding to the minimum distance to target coordinates
        """
        # Calculate squared distance to target for every grid point
        distance = self.haversine_distance(ds.lat, ds.lon, target_lat, target_lon)

        # Find the 2D indices of the minimum distance
        min_dist_idx = distance.argmin(dim=['y', 'x'])

        y_idx = int(min_dist_idx['y'])
        x_idx = int(min_dist_idx['x'])

        return x_idx, y_idx

    def find_coordinates_index(self):
        """
            Obtain the indexes x and y of the SAR-measured image closest to self.lat and self.lon.
            Due to the large size of the images, a recursive search is carried out, decimating the image at each step
            for faster convergence.        
        """
        # Get the image sizes
        x_max_len = self.ds.sizes['x']
        y_max_len = self.ds.sizes['y']

        # Coarse Search (decimate by a factor 1000)
        skip_1 = 1000
        ds_step1 = self.ds.isel(x=slice(0, None, skip_1), y=slice(0, None, skip_1))
        x_idx, y_idx = self.find_minimum_distance(ds_step1, self.lat, self.lon)
        
        # Scale back to full dataset coordinates
        x_abs = x_idx * skip_1
        y_abs = y_idx * skip_1

        # Define bounded box around the coarse estimate
        x_min = max(0, x_abs - skip_1)
        x_max = min(x_max_len, x_abs + skip_1 + 1)
        y_min = max(0, y_abs - skip_1)
        y_max = min(y_max_len, y_abs + skip_1 + 1)

        # Medium Search (decimate by a factor 100)
        skip_2 = 100
        ds_step2 = self.ds.isel(x=slice(x_min, x_max, skip_2), y=slice(y_min, y_max, skip_2))
        x_idx, y_idx = self.find_minimum_distance(ds_step2, self.lat, self.lon)
        
        # Map back to full dataset coordinates
        x_abs = x_min + (x_idx * skip_2)
        y_abs = y_min + (y_idx * skip_2)

        # Define fine bounded box
        x_min = max(0, x_abs - skip_2)
        x_max = min(x_max_len, x_abs + skip_2 + 1)
        y_min = max(0, y_abs - skip_2)
        y_max = min(y_max_len, y_abs + skip_2 + 1)

        # Fine Search (full resolution)
        ds_reduced = self.ds.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
        x_idx, y_idx = self.find_minimum_distance(ds_reduced, self.lat, self.lon)
        
        # Final global indices
        final_x = x_min + x_idx
        final_y = y_min + y_idx

        self.x_coordinate = final_x
        self.y_coordinate = final_y

    def obtain_target_tile(self):
        """
            Function to retrieve a tile of SAR-measured values at the target coordinates of the specified width and height at the class constructor
        
        Return:
        self.tile : xr.Dataset
            Dataset containing the target tile of the image.
        """
        self.find_coordinates_index()

        self.tile = self.ds.sel(x=slice(self.x_coordinate-int(self.width/2),self.x_coordinate+int(self.width/2)),y=slice(self.y_coordinate-int(self.height/2),self.y_coordinate+int(self.height/2)))

        return self.tile

    def cfar_mask_2d_numpy(self, arr, num_guard=3, num_ref=6, pfa=1e-4):
        """
        NumPy 2D CA-CFAR implementation operating on spatial arrays for target detection.

        Params:
        arr : xr.Array
            Input SAR image array
        num_guard : integer
            guard pixels for the filter
        num_ref : integer
            reference pixels for the filter
        pfa : float
            Probability of false alarm for the filter.

        Return:
        arr_clean : xr.Array
            Array after removal of objects detected by the CA-CFAR filter
        target_mask : xr.Array
            Mask with the identified objects 
        """
        nan_mask = np.isnan(arr)
        arr_filled = np.nan_to_num(arr, nan=0.0).astype(np.float64)

        guard_size = 2 * num_guard + 1
        total_size = 2 * (num_guard + num_ref) + 1
        n_ref_cells = (total_size**2) - (guard_size**2)

        # Scipy uniform filter to compute local reference sums
        sum_total = uniform_filter(arr_filled, size=total_size) * (total_size**2)
        sum_guard = uniform_filter(arr_filled, size=guard_size) * (guard_size**2)

        # Calculate noise power and dynamic threshold factor (alpha)
        noise_power = (sum_total - sum_guard) / n_ref_cells
        alpha = n_ref_cells * (pfa**(-1.0 / n_ref_cells) - 1.0)
        threshold = alpha * noise_power

        # Target detection decision
        target_mask = (arr_filled > threshold) & (~nan_mask)

        arr_clean = arr.copy()
        arr_clean[target_mask] = np.nan

        return arr_clean, target_mask


    def filter_objects(self, var_name='Sigma0_VV', spatial_dims=None, 
                            num_guard=20, num_ref=20, pfa=1e-3):
        """
        Apply CFAR Target Masking directly to the previously cropped tile.
        Auto-detects spatial dimensions if spatial_dims is None.

        Params:
        var_name : String
            Name of the variable to be processed by the object detection filter removal.
        spatial_dims : Tuple of strings
            Name of the spatial dimensions of the image.
        num_guard : integer
            guard pixels for the filter
        num_ref : integer
            reference pixels for the filter
        pfa : float
            Probability of false alarm for the filter.    

        Return:
        clean_ds : xr.Dataset
            The dataset containing the tile with the objects removed.   
        """
        da = self.tile[var_name] if isinstance(self.tile, xr.Dataset) else self.tile

        # Auto-detect spatial dimensions in exact dataset order
        if spatial_dims is None:
            spatial_dims = da.dims[-2:]
        
        # Ensure spatial_dims match the exact order in the DataArray
        spatial_dims = tuple(d for d in da.dims if d in spatial_dims)

        def _ufunc_wrapper(arr):
            cleaned, mask = self.cfar_mask_2d_numpy(
                arr, num_guard=num_guard, num_ref=num_ref, pfa=pfa
            )
            return cleaned, mask

        # Execute ufunc across x and y spatial dimensions
        da_clean, da_mask = xr.apply_ufunc(
            _ufunc_wrapper,
            da,
            input_core_dims=[list(spatial_dims)],
            output_core_dims=[list(spatial_dims), list(spatial_dims)],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[da.dtype, bool]
        )

        da_clean.name = f"{var_name}_no_targets"
        da_mask.name = "target_mask"

        if isinstance(self.tile, xr.Dataset):
            ds_out = self.tile.copy()
            ds_out[da_clean.name] = da_clean
            ds_out[da_mask.name] = da_mask
            self.tile = ds_out
            return ds_out
        else:
            return xr.merge([da_clean, da_mask])

    def compute_fft_2D(self, var_name='Sigma0_VV_no_targets', spatial_dims=('y', 'x'), filter=True, normalize=True):
            """
            Computes 2D FFT on an xarray DataArray and returns a DataArray with frequency axes.
    
            Params:
            var_name : String
                Name of the variable to be processed by the object detection filter removal.
            spatial_dims : Tuple of strings
                Name of the spatial dimensions of the image.
            
            Returns:
            fft_da : xr.Array
                The computed PSD with the periodogram method
            """
            dim_y, dim_x = spatial_dims
            
            # Fill NaNs before FFT (FFTs cannot process NaNs)
            da_filled = self.tile[var_name].fillna(np.mean(self.tile[var_name]))
    
            # Subtract mean component
            da_demeaned = da_filled - np.mean(da_filled)

            # Get the signal energy
            total_energy_spatial = np.sum(da_demeaned.values**2)

            # Calculate sampling intervals (dx, dy) in physical units
            dy = np.abs(np.diff(self.tile[dim_y].values)[0])
            dx = np.abs(np.diff(self.tile[dim_x].values)[0])
    
            # Assumed values
            dy = self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing']
            dx = self.tile.metadata.attrs['Abstracted_Metadata:range_spacing']
    
            if filter == True:
                # 3. Create and apply a 2D Hanning window to suppress boundary artifacts
                Ny, Nx = da_demeaned.shape
                win_y = np.hanning(Ny)
                win_x = np.hanning(Nx)
                window_2d = np.outer(win_y, win_x)
                
                da_demeaned = da_demeaned * window_2d
    
            # 2D FFT computation
            fft_vals = np.fft.fftshift(np.fft.fft2(da_demeaned))
            psd_raw = np.abs(fft_vals)**2
    
            # Normalize PSD to the maximum value if desired
            if normalize == True:
                psd_max = np.max(psd_raw)
                if psd_max > 0:
                    psd_norm = psd_raw / psd_max
                else:
                    psd_norm = psd_raw
                    
                self.tile['fft_normalization_factor'] = psd_max
            else:
                psd_norm = psd_raw
    
            # Convert to dB (Normalized max will be 0 dB)
            power_db = 10 * np.log10(psd_norm + 1e-8)
            
            # Compute centered frequency coordinates (cycles per spatial unit)
            freq_y = np.fft.fftshift(np.fft.fftfreq(self.tile[dim_y].size, d=dy))
            freq_x = np.fft.fftshift(np.fft.fftfreq(self.tile[dim_x].size, d=dx))
            
            # Construct output DataArray
            fft_da = xr.DataArray(
                power_db,
                coords={'freq_y': freq_y, 'freq_x': freq_x},
                dims=['freq_y', 'freq_x'],
                name='power_spectrum_db'
            )
            fft_da.freq_y.attrs['units'] = '1/m'
            fft_da.freq_x.attrs['units'] = '1/m'
    
            self.psd = fft_da
            self.tile['spatial_energy'] = total_energy_spatial
            
            return fft_da
    

    def compute_welch_2D(self, var_name='Sigma0_VV_no_targets', spatial_dims=('y', 'x'), tile_size=(256, 256), overlap=0.5, window='hanning', return_db=True):
        """
        Computes 2D Welch Power Spectral Density (PSD) on an xarray DataArray 
        and returns a DataArray with centered spatial frequency axes (cycles per unit) using the Welch method.

        Params:
        da : xarray.DataArray
            Input 2D SAR spatial DataArray.
        spatial_dims : tuple of str
            The 2D spatial dimension names, e.g., ('y', 'x') or ('lat', 'lon').
        tile_size : tuple of int (ty, tx)
            Size of the sliding sub-tiles in pixels (default 256x256).
        overlap : float (0.0 to 1.0)
            Fractional overlap between adjacent tiles (0.5 = 50% overlap).
        window : str
            Type of 2D windowing function ('hanning', 'hamming', or 'boxcar').
        return_db : bool
            If True, converts PSD to decibels: 10 * log10(PSD).

        Returns:
        xr.DataArray
            2D Welch PSD with centered frequency coordinates (freq_y, freq_x).
        """
        dim_y, dim_x = spatial_dims
        
        # 1. Fill NaNs with mean to prevent FFT corruption
        da_filled = self.tile[var_name].fillna(self.tile[var_name].mean().item()) - np.mean(self.tile[var_name].fillna(self.tile[var_name].mean().item()))
        arr = da_filled.values.astype(np.float64)
        
        ny, nx = arr.shape
        ty, tx = tile_size

        # 2. Extract spatial resolution (dx, dy) from coordinate vectors
        dy = np.abs(np.diff(self.tile[dim_y].values)[0])
        dx = np.abs(np.diff(self.tile[dim_x].values)[0])
        
        # Assumed values
        dy = self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing']
        dx = self.tile.metadata.attrs['Abstracted_Metadata:range_spacing']

        # 3. Calculate step sizes based on overlap percentage
        step_y = max(1, int(ty * (1.0 - overlap)))
        step_x = max(1, int(tx * (1.0 - overlap)))

        # 4. Construct 2D tapering window and energy normalization factor
        if window == 'hanning':
            win_2d = np.outer(np.hanning(ty), np.hanning(tx))
        elif window == 'hamming':
            win_2d = np.outer(np.hamming(ty), np.hamming(tx))
        else:
            win_2d = np.ones((ty, tx))
            
        win_norm = np.sum(win_2d**2) * (dx * dy)  # Physical scale normalization

        # 5. Tile iteration and 2D FFT averaging
        psd_accumulator = np.zeros((ty, tx), dtype=np.float64)
        tile_count = 0

        for y in range(0, ny - ty + 1, step_y):
            for x in range(0, nx - tx + 1, step_x):
                tile = arr[y:y+ty, x:x+tx]
                
                # Remove DC offset per tile
                tile_detrend = tile #- np.mean(tile)
                
                # Apply 2D spatial window
                tile_windowed = tile_detrend * win_2d
                
                # 2D FFT & Power calculation
                fft2_tile = np.fft.fft2(tile_windowed)
                power_tile = (np.abs(fft2_tile)**2) / win_norm
                
                psd_accumulator += power_tile
                tile_count += 1

        if tile_count == 0:
            raise ValueError(f"Tile size {tile_size} is larger than input array shape ({ny}, {nx})!")

        # 6. Average across tiles and shift zero-frequency (DC) component to center
        psd_avg = psd_accumulator / tile_count
        psd_shifted = np.fft.fftshift(psd_avg)

        # Convert to dB scale if requested
        if return_db:
            psd_output = 10 * np.log10(psd_shifted + 1e-12)
            var_name = 'welch_psd_db'
            unit_label = 'dB'
        else:
            psd_output = psd_shifted
            var_name = 'welch_psd'
            unit_label = f"({self.tile_filtered.attrs.get('units', 'intensity')})^2 m^2"

        # 7. Calculate centered frequency axes (cycles per spatial unit)
        freq_y = np.fft.fftshift(np.fft.fftfreq(ty, d=dy))
        freq_x = np.fft.fftshift(np.fft.fftfreq(tx, d=dx))

        # 8. Build xarray.DataArray
        psd_da = xr.DataArray(
            psd_output,
            coords={'freq_y': freq_y, 'freq_x': freq_x},
            dims=['freq_y', 'freq_x'],
            name=var_name
        )
        
        # Preserve metadata attributes
        psd_da.attrs['units'] = unit_label
        psd_da.freq_y.attrs['units'] = '1/m'
        psd_da.freq_x.attrs['units'] = '1/m'
        psd_da.attrs['tile_size'] = f"{ty}x{tx}"
        psd_da.attrs['overlap'] = f"{int(overlap*100)}%"
        psd_da.attrs['num_tiles_averaged'] = tile_count

        self.psd = psd_da
        
        return psd_da

    def compute_autocorr_2D(self, var_name="Sigma0_VV_no_targets", spatial_dims=("y", "x"), filter=True, normalize=True):
        """Computes 2D Autocorrelation on an xarray DataArray via FFT and returns

        a DataArray with physical spatial lag coordinates (in meters).

        Params:
        var_name : String
            Name of the variable to be processed.
        spatial_dims : Tuple of strings
            Name of the spatial dimensions of the image.
        filter : Bool
            Whether to apply a 2D Hanning window before computing autocorrelation.
        normalize : Bool
            If True, normalizes the autocorrelation peak at zero-lag to 1.0.

        Returns:
        autocorr_da : xr.DataArray
            The 2D autocorrelation surface indexed by spatial lags (lag_y, lag_x).
        """
        dim_y, dim_x = spatial_dims

        # Fill NaNs before FFT operations (FFTs cannot process NaNs)
        da_filled = self.tile[var_name].fillna(np.mean(self.tile[var_name]))

        # Subtract mean component (Crucial for autocorrelation to isolate spatial structures)
        da_demeaned = da_filled - np.mean(da_filled)

        # Spatial dimensions
        Ny, Nx = da_demeaned.shape

        # Retrieve sampling intervals (dy, dx) in meters from metadata
        dy = self.tile.metadata.attrs["Abstracted_Metadata:azimuth_spacing"]
        dx = self.tile.metadata.attrs["Abstracted_Metadata:range_spacing"]

        if filter:
            # Apply a 2D Hanning window to suppress boundary discontinuity artifacts
            win_y = np.hanning(Ny)
            win_x = np.hanning(Nx)
            window_2d = np.outer(win_y, win_x)

            da_demeaned = da_demeaned * window_2d

        # --- Wiener–Khinchin Theorem for 2D Autocorrelation ---
        # 1. Zero-pad to avoid circular correlation / aliasing effects
        pad_shape = (2 * Ny - 1, 2 * Nx - 1)

        # 2. FFT -> Power Spectral Density (PSD) -> Inverse FFT
        fft_vals = np.fft.fft2(da_demeaned, s=pad_shape)
        psd_raw = np.abs(fft_vals) ** 2
        autocorr_raw = np.fft.ifft2(psd_raw).real

        # 3. Shift zero-lag (maximum correlation peak) to center
        autocorr_shifted = np.fft.fftshift(autocorr_raw)

        # Normalize autocorrelation so zero-lag peak = 1.0
        if normalize:
            zero_lag_val = autocorr_shifted[
                autocorr_shifted.shape[0] // 2, autocorr_shifted.shape[1] // 2
            ]
            if zero_lag_val > 0:
                autocorr_out = autocorr_shifted / zero_lag_val
            else:
                autocorr_out = autocorr_shifted
            self.tile["autocorr_peak_value"] = zero_lag_val
        else:
            autocorr_out = autocorr_shifted

        # --- Construct Physical Spatial Lag Axes (in meters) ---
        # Centered lags ranging from -(N-1)*d to +(N-1)*d
        lags_y = np.arange(-(Ny - 1), Ny) * dy
        lags_x = np.arange(-(Nx - 1), Nx) * dx

        # Construct output DataArray matching the xarray workflow
        autocorr_da = xr.DataArray(
            autocorr_out,
            coords={"lag_y": lags_y, "lag_x": lags_x},
            dims=["lag_y", "lag_x"],
            name="autocorrelation",
        )
        autocorr_da.lag_y.attrs["units"] = "m"
        autocorr_da.lag_x.attrs["units"] = "m"

        # Save to object attribute and return
        self.autocorr = autocorr_da

        return autocorr_da

    def get_closest_measurement(self, datetime_col='TIME'):
        """
        Finds the row in FINO1 dataframe self.df where datetime_col is closest to target_time_str.
        Requires that the FINO1 dataframe is read with self.read_fino_file()

        Params:
        datetime_col : String
            Name of the datetime column in FINO1 dataset
        
        Returns:
        self.tile : xr.Dataset
            The cropped SAR tile with the added FINO1 information corresponding to the SAR measurement time.
        """
        target_time_str = self.tile.start_date

        # Parse target string into a pandas Timestamp
        target_dt = pd.to_datetime(target_time_str)
        
        # Ensure the dataframe column is datetime type
        dt_series = pd.to_datetime(self.df[datetime_col])
        
        # Find index of minimum absolute time difference
        closest_idx = (dt_series - target_dt).abs().idxmin()
        
        # Return the row as a DataFrame (or Series via df.loc[closest_idx])
        row = self.df.loc[[closest_idx]]
    
        # Convert the single row into a Series
        row_series = row.iloc[0]
    
        # Add each item from the row into your Dataset
        for col_name, value in row_series.items():
            # If the value is a pandas Timestamp, convert to string or numpy datetime
            if isinstance(value, pd.Timestamp):
                value = str(value)
                
            self.tile[col_name] = value
    
        return self.tile

    ########### TODO: functions to implement ####################

    # def retrieve_stability_from_psd(self):

    def wind_retrieval(self,tile, lat:float, lon:float , var_name='Sigma0_VV_no_targets', v_bounds = (0.5, 40.0),
        wdir:float = 180.0, n_v_starts: int = 4, wind_direction_convention='from', level:int = 1, 
        weighting: str = None, gaussian_sigma_m: float = 100.0):
        # weigthing: None | "uniform" | "gaussian" | "inverse_distance2"
        R = 6_371_000.0  # Earth radius in meters
        # 1. Find the SAR pixel closest to the requested lat/lon
        sigma0_da = tile[var_name] # Sigma0_VV_no_targets is  filtered tile with no obstacles
        if sigma0_da.ndim != 2: raise ValueError(f"{var_name} must be 2-D. Got shape {sigma0_da.shape}")

        # SAR geographical coordinates
        # Retrieve lat and lon for each SAR pixel a put in an array
        lat_da = tile['lat']
        lon_da = tile['lon']
        lat_array = np.asarray(lat_da.values, dtype=np.float64)
        lon_array = np.asarray(lon_da.values, dtype=np.float64)
        if lat_array.shape != sigma0_da.shape or lon_array.shape != sigma0_da.shape: raise ValueError("lat/lon and sigma0 must have the same spatial shape.")

        # Find closest pixel
        lat0_rad = np.deg2rad(lat)
        dx = R * np.cos(lat0_rad) * np.deg2rad(lon_array - lon)
        dy = R * np.deg2rad(lat_array - lat)
        distance2_m = dx**2 + dy**2
        invalid = ~np.isfinite(lat_array) | ~np.isfinite(lon_array)
        distance2_m[invalid] = np.inf
        # distance2 = (lat_array - lat)**2 + (lon_array - lon)**2 # TODO: here we already have the distance to each pixel respecto to FINO1
        distance2 = distance2_m
        y_idx, x_idx = np.unravel_index(np.argmin(distance2), distance2.shape)

        # ---------------------------------------------
        # Window around FINO1
        # level=1 -> 3x3
        # level=N -> (2*level+1)^2
        # ---------------------------------------------

        if level < 1: raise ValueError("level must be >= 1")

        y_min = y_idx - level
        y_max = y_idx + level + 1
        x_min = x_idx - level
        x_max = x_idx + level + 1
        ny, nx = sigma0_da.shape
        y_min = max(0, y_min)
        y_max = min(ny, y_max)
        x_min = max(0, x_min)
        x_max = min(nx, x_max)

        if y_min < 0 or y_max > ny or x_min < 0 or x_max > nx: raise ValueError(f"Window level={level} around pixel ({y_idx}, {x_idx}) falls outside the SAR image.")

        sigma0_window = np.asarray(sigma0_da.values[y_min:y_max, x_min:x_max],dtype=np.float64)
        # ---------------------------------------------
        # Distance-based weights
        # ---------------------------------------------
        distance2_window = distance2[y_min:y_max, x_min:x_max]
        valid = (np.isfinite(distance2_window) & np.isfinite(sigma0_window))

        if weighting is None:
            # No weighting: ordinary least squares
            weights_window = np.ones_like(sigma0_window, dtype=np.float64)

        elif weighting == "uniform":
            # Flat distribution, normalized to sum to 1
            weights_window = valid.astype(np.float64)
            weight_sum = weights_window.sum()
            if weight_sum == 0: raise ValueError("No valid pixels available.")
            weights_window /= weight_sum
        elif weighting == "inverse_distance2":
            weights_window = np.zeros_like(sigma0_window, dtype=np.float64)
            weights_window[valid] = 1.0 / np.maximum(distance2_window[valid], 1.0)

        elif weighting == "gaussian":
            if gaussian_sigma_m <= 0: raise ValueError("gaussian_sigma_m must be positive.")
            # Gaussian weights based on distance to FINO1.
            d2_min = np.min(distance2_window[valid]) if valid.any() else np.inf
            if not np.isfinite(d2_min): raise ValueError("No valid pixels available.")
            exponent = -(distance2_window - d2_min) / (2.0 * gaussian_sigma_m**2)
            weights_window = np.zeros_like(sigma0_window, dtype=np.float64)
            weights_window[valid] = np.exp(exponent[valid])
            weights_window /= weights_window.sum()

        else: raise ValueError("weighting must be None, 'uniform' or 'gaussian'.")

        print("=================================")
        print("Wind retrieval window")
        print("level =", level)
        print("window shape =", sigma0_window.shape)
        print("center pixel =", (y_idx, x_idx))
        
        # 2. Get incidence angle at the same position
        theta_da = tile['incident_angle']
        theta_tp = np.asarray(theta_da.values, dtype=np.float64)
        if theta_tp.ndim != 2:
            raise ValueError("incident_angle must be a 2-D tie-point grid.")
        tp_ny, tp_nx = theta_tp.shape

        if (tp_ny, tp_nx) != (11, 11):
            print(f"Warning: expected an 11x11 incidence-angle grid, got {theta_tp.shape}")

        # Coordinates of the tie-point grid in pixel coordinates
        ny, nx = sigma0_da.shape
        tp_y = np.linspace(0.0, ny - 1.0, tp_ny)
        tp_x = np.linspace(0.0, nx - 1.0, tp_nx)

        interpolator = RegularGridInterpolator((tp_y, tp_x), theta_tp, method='linear', bounds_error=False, fill_value=None)
        
        # Pixel coordinates of the window
        window_y, window_x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        points = np.column_stack([window_y.ravel(), window_x.ravel()])
        theta_window = interpolator(points).reshape(sigma0_window.shape)
        print("theta_window shape =", theta_window.shape)

        # 3. Platform heading (satellite fligh heading)
        heading_raw = tile['platform_heading'].values

        try:
            platform_heading = float(np.asarray(heading_raw).item())
        except (TypeError, ValueError):
            raise ValueError(f"Could not convert platform_heading={heading_raw!r} to float.")

        # Normalize to [0, 360)
        platform_heading %= 360.0

        # Sentinel-1 is not right-looking.
        # For a right-looking SAR --> look azimuth = platform heading + 90° # TODO
        look_azimuth = platform_heading % 360.0
        print("look_azimuth 00: ", look_azimuth)

        # 5. Wind direction convention
        if wind_direction_convention not in ('from', 'to'):
            raise ValueError("wind_direction_convention must be 'from' or 'to'.")
        
        #Flat values for residuals
        sigma0_flat = sigma0_window.ravel()
        theta_flat = theta_window.ravel()
        weights_flat = weights_window.ravel()
        
        valid_mask = (np.isfinite(sigma0_flat) & np.isfinite(theta_flat) & np.isfinite(weights_flat) & (sigma0_flat > 0))
        sigma0_flat = sigma0_flat[valid_mask]
        theta_flat = theta_flat[valid_mask]
        weights_flat = weights_flat[valid_mask]

        if sigma0_flat.size < 3:
            raise ValueError(f"Only {sigma0_flat.size} valid pixels in level={level} window.")
        weights_flat /= np.mean(weights_flat)
        sqrt_weights = np.sqrt(weights_flat)

        # 6. CMOD5-N residual function
        def residuals(params):
            v = float(params[0])
            if wind_direction_convention == 'from':
                wind_to = (wdir + 180.0) % 360.0 # Normalize direction.
            else:
                wind_to = wdir % 360.0 # Normalize direction.

            phi_rel = (wind_to - look_azimuth) # relative angle between wind direction and satellite heading

            phi_rel = (phi_rel + 180.0) % 360.0 - 180.0 # [-180°, 180°) TODO: why?

            # CMOD5-N
            sigma0_pred = calc_sigma0_cmod5_n(v, phi_rel, theta_flat) #Using empirical equation
            # Guard against numerical problems.
            sigma0_pred = np.asarray(sigma0_pred, dtype=np.float64)

            sigma0_pred = np.maximum(sigma0_pred, 1e-12)

            res = sigma0_pred - sigma0_flat
            return sqrt_weights*res
        
        # 8. Multi-start optimization
        v_min, v_max = v_bounds

        v_grid = np.linspace(v_min, v_max, n_v_starts)

        best_result = None
        best_cost = np.inf
        all_results = dict()
        iteration = 0
        for v0 in v_grid:
            try:
                result = least_squares(
                    residuals,
                    x0=[v0],
                    bounds=([v_min], [v_max]),
                    method='trf',
                    max_nfev=300,
                )

                if result.success and np.isfinite(result.cost):
                    v_ret = float(result.x[0])
                    # phi_ret = float(result.x[1] % 360.0)

                    all_results[iteration] = {
                        # 'result': result,
                        'phi0': float(wdir),
                        'v0': float(v0),
                        'v_ret': v_ret,
                        'cost': float(result.cost),
                    }

                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_result = result

                    iteration += 1

            except Exception as exc:
                print(f"Error para v0={v0}: {type(exc).__name__}: {exc}")
                continue
                
        # 8. Check optimization
        if best_result is None:
            print( "wind_retrieval: all optimization attempts failed.")
            self.retrieved_wind_speed = np.nan
            return np.nan, tile, all_results

        # 9. Retrieve solution
        v_ret = float(best_result.x[0])
        # phi_ret = float(best_result.x[1] % 360.0)

        # 10. Store results
        self.retrieved_wind_speed = v_ret
        tile['retrieved_wind_speed'] = v_ret
        tile['wind_retrieval_cost'] = float(best_cost)
        tile['platform_heading_used'] = float(platform_heading)
        tile['radar_look_azimuth_used'] = float(look_azimuth)
        self.tile = tile
        return v_ret, tile, all_results

    def retrieve_wind_profile_parameters(self, wind_speeds: list[float], heights: list[float], weights: list[float],
        obukhov_b, u_star_bounds, n_starts: float, stability, z_ref: float,
        delta_v: bool = False, scaled: bool = False):
        
        # 1. Convert inputs to arrays
        wind_speeds = np.asarray(wind_speeds, dtype=float).ravel()
        heights = np.asarray(heights, dtype=float).ravel()

        if delta_v:
            heights_combination, wind_speeds = self.calculate_differences(heights, wind_speeds)
        
        if weights is None or delta_v:
            weights = np.ones_like(wind_speeds)

        # 3. Validate friction velocity bounds
        u_min, u_max = map(float, u_star_bounds)

        # 4. Validate Obukhov length bounds
        obu_min, obu_max = map(float, obukhov_b)

        # 5. Define residual function
        def residuals(params):
            # Original equation not scaled
            if not delta_v and not scaled:
                u_star, L = params
                wind_pred = diabatic_wind_profile(heights, u_star, L, stability)
            
            # Original equation scaled
            if not delta_v and scaled:
                u_star, eta = params
                wind_pred = diabatic_wind_profile(heights, u_star, eta, z_ref=z_ref, scaled=scaled, stability=stability)
            
            #Diff equation not scaled
            if delta_v and not scaled:
                u_star, L = params
                wind_pred = delta_diabatic_wind_profile(z2=heights_combination[:,0], z1=heights_combination[:,1], 
                                                        u_star=u_star, obu_parameter=L, stability=stability)
            
            #Diff equation not scaled
            if delta_v and scaled:
                u_star, eta = params
                wind_pred = delta_diabatic_wind_profile(z2=heights_combination[:,0], z1=heights_combination[:,1], 
                                                        u_star=u_star, obu_parameter=eta, z_ref=z_ref, scaled=scaled, stability=stability)
                            
            r_data = np.sqrt(weights) * (wind_pred - wind_speeds)
            return r_data

        # 7. Multi-start optimization
        u_grid = np.linspace(u_min, u_max, n_starts)
        obu_grid = np.linspace(obu_min, obu_max, n_starts)

        best_result = None
        best_cost = np.inf
        all_results = {}
        i = 0
        for u0 in u_grid:
            for obu0 in obu_grid:
                result = least_squares(
                    residuals,
                    x0=[u0, obu0],
                    bounds=(
                        [u_min, obu_min],
                        [u_max, obu_max],
                    ),
                    method="trf",
                    x_scale=[0.1, max(1.0, abs(obu0)) if not scaled else max(0.01, abs(obu0))],
                    max_nfev=1000,
                )

                all_results[i] = {
                    "u_star_0": float(u0),
                    "L_0": float(obu0),
                    "u_star_ret": float(result.x[0]),
                    "L_ret": float(result.x[1]) if not scaled else float(z_ref/result.x[1]),
                    "cost": float(result.cost),
                    "success": bool(result.success),
                }
                i += 1

                if result.success and np.isfinite(result.cost) and result.cost < best_cost:
                    best_cost = float(result.cost)
                    best_result = result

        if best_result is None: raise RuntimeError("All optimization attempts failed.")
        
        u_star_ret = float(best_result.x[0])
        L_ret = float(best_result.x[1]) if not scaled else float(z_ref/best_result.x[1])

        best_info = {
            "cost": best_cost,
            "success": bool(best_result.success),
            "optimality": float(best_result.optimality),
            "nfev": int(best_result.nfev),
            "stability": stability,
        }

        return L_ret, u_star_ret, best_info, all_results

    def retrieve_wind_profile_parameters_all_stability(self, wind_speeds, heights, weights=None,
        u_star_bounds=(0.01, 2.0), n_starts: int = 5, z_ref: float = 60.0, 
        delta_v: bool = False, scaled: bool = False):
        
        fits = self.return_bounds(z_ref, scaled)
        results = {}

        for stability, obukhov_b in fits:
            L_ret, u_star_ret, best_info, all_results = self.retrieve_wind_profile_parameters(
                wind_speeds=wind_speeds, heights=heights, weights=weights,
                obukhov_b=obukhov_b, u_star_bounds=u_star_bounds, z_ref=z_ref,
                n_starts=n_starts, stability=stability, delta_v=delta_v, scaled=scaled)
            results[stability] = {
                "L": float(L_ret),
                "u_star": float(u_star_ret),
                "cost": float(best_info["cost"]),
                "success": bool(best_info["success"]),
            }

        return results
    
    @staticmethod
    def calculate_differences(heights: list[float], velocity: list[float]):
        heights = np.asarray(heights)
        velocity = np.asarray(velocity)

        max_idx = np.argmax(heights)
        # min_idx = np.argmin(heights)

        height_pairs = []
        velocity_diffs = []

        for i in range(len(heights)):
            if i != max_idx:
                height_pairs.append([heights[max_idx], heights[i]])
                velocity_diffs.append(velocity[max_idx] - velocity[i])
            # if i != min_idx:
            #     height_pairs.append([heights[i], heights[min_idx]])
            #     velocity_diffs.append(velocity[i] - velocity[min_idx])
        return np.array(height_pairs), np.array(velocity_diffs)
    
    @staticmethod
    def return_bounds(z_ref: float = 60.0, scaled: bool = False):
        low_level = 10.0
        intermediate_level = 200.0
        upper_level = 500.0
        
        if scaled:
            fits = [
                ("u", (z_ref/-low_level, z_ref/-intermediate_level)),
                ("nu", (z_ref/-intermediate_level, z_ref/-upper_level)),
                ("ns", (z_ref/upper_level, z_ref/intermediate_level)),
                ("s", (z_ref/intermediate_level, z_ref/low_level))
            ]
        else:
            fits = [
                ("u", (-intermediate_level, -low_level)),
                ("nu", (-upper_level, -intermediate_level)),
                ("ns", (intermediate_level, upper_level)),
                ("s", (low_level, intermediate_level)),
            ]
        return fits