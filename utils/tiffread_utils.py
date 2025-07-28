
from typing import Union, Any, Dict
from pathlib import Path
import numpy as np

from osgeo import gdal
gdal.UseExceptions()


def load_file(filename: str, folder: str = None, verbose: bool = False) -> Dict[str, Any]:
    """ 
    Reads bathymetry data from a TIFF file.

    This function uses the GDAL library to read a TIFF file
    containing bathymetric data. It extracts the primary data matrix,
    the "no-data" value (NDV), and the spatial resolution of the image.

    Parameters
    ----------
    filename : str
        The base name of the TIFF file (e.g., "bathymetry.tif").
    folder : str, optional
        An optional path to the directory containing the TIFF file.
        If provided, the function will look for `filename` inside this `folder`.
        If None, `filename` is assumed to be an absolute path or relative
        to the current working directory.
    verbose : bool
              Optional flag to print some information on the bathymetry file

    Returns
    -------
    dict
        A dictionary containing the parsed bathymetric data:
        * 'depth' (numpy.ndarray of float): A 2D NumPy array containing the
            bathymetric depth values in meters. Pixels with no data are
            represented by the 'ndv' value.
        * 'ndv' (float or int or None): The value representing "no-data" pixels
            in the `depth` array. This can be `None`, `np.nan`, or a specific
            numeric value as defined in the TIFF file's metadata.
        * 'resolution' (float): The spatial resolution of the bathymetric data
            in meters per pixel (derived from the GeoTransform).
        * 'dimensions' (tuple) : Dimensions of the 2D NumPy array
            

    Raises
    ------
    TypeError
        If `filename` is not a string. (Changed from ValueError for type-checking)
    FileNotFoundError
        If the specified TIFF file does not exist at the given path.
    RuntimeError
        If GDAL fails to open the file (e.g., corrupt file, unsupported format)
        or if the primary raster band cannot be retrieved.
    ValueError
        If spatial resolution is less than or equal to zero

    See Also
    --------
    osgeo.gdal.Open : GDAL's function for opening raster datasets.
    os.path.join : For robust path construction across operating systems.

    Notes
    -----
    This function relies on the `osgeo.gdal` library, which must be installed
    and correctly configured in your environment.

    Examples
    --------
    >>> # Assuming 'data/my_bathymetry.tif' exists
    >>> import numpy as np
    >>> # Example 1: File in current directory
    >>> data = load_file("my_bathymetry.tif")
    >>> print(data['resolution'])
    1.0
    >>> # Example 2: File in a specific folder
    >>> data_folder = "data"
    >>> data = load_file("another_map.tif", folder=data_folder)
    >>> print(data['depth'].shape)
    (100, 200)
    >>> # Example 3: Handling no-data values
    >>> ndv_value = data['ndv']
    >>> if ndv_value is not None:
    >>>     print(f"No-data value: {ndv_value}")
    >>>     # Count no-data pixels
    >>>     print(f"No-data pixels: {np.sum(data['depth'] == ndv_value)}")
    
    """
    
    if not isinstance(filename, str):
        # Using TypeError for incorrect type, more conventional than ValueError
        raise TypeError("Filename must be of type string.")

    if folder:
        file_path = Path(folder) / filename
    else:
        file_path = filename
    
    # Check if the file exists before trying to open with GDAL
    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found at: {str(file_path)}")

    ds = gdal.Open(str(file_path)) # Ensure path is string for gdal.Open
    if not ds:
        raise RuntimeError(f"GDAL failed to open TIFF file: '{file_path}'. It might be corrupt, an unsupported format, or a permission issue.")

    depth_band = ds.GetRasterBand(1)
    if not depth_band:
        # This could happen if band 1 doesn't exist or is invalid
        raise RuntimeError(f"Could not retrieve raster band 1 from TIFF file: '{file_path}'. File might be empty or malformed.")

    ndv = depth_band.GetNoDataValue()
    depth = depth_band.ReadAsArray()

    depth_gt = ds.GetGeoTransform()
    resolution = depth_gt[1] # GDAL GeoTransform has resolution at index 1 (pixel width)
    if resolution <= 0:
        raise ValueError(f"Spatial resolution is less than or equal to zero: {resolution}")

    # Free up memory
    ds = None 
    
    if verbose:
        # Print some statistics of the bathymetry data
        print(f"""Input filename: {str(file_path)}
              Data dimensions: {depth.shape}
              Min/Max: {np.nanmin(depth), np.nanmax(depth)}
              Survey Coverage: {int(depth.shape[0] * resolution)}m x {int(depth.shape[0] * resolution)}m
              Spatial Resolution: {resolution}
              """)

    # Compile bathymetric data in a dictionary and return directly
    return {
        'depth': depth,
        'ndv': ndv,
        'resolution': resolution,
        'dimensions': depth.shape
    }
    
    
def remove_edge_Nans(depth: np.array, ndv: Union[float, int, None], 
        max_iterations: int = None, verbose: bool = False) -> np.array:
    
    """ 
    Iteratively removes rows and columns containing no-data values from the edges.

    This function crops the input 2D array by iteratively removing its outermost
    rows and columns if they contain any pixels matching the specified "no-data"
    value (NDV). The process continues until all edges are free of NDV pixels
    or a maximum iteration limit is reached.

    Parameters
    ----------
    depth : numpy.ndarray
        A 2D NumPy array representing surface elevation or bathymetry data.
        Expected to be numeric (e.g., float, int).
    ndv : float or int or None
        The value representing "no-data" pixels within the `depth` array.
        This could be a specific number (e.g., -9999), `np.nan`, or `None`.
    verbose : bool, optional
        If True, print detailed information about the cropping process,
        including original and new data dimensions and the number of layers removed.
        Defaults to False.
    max_iteration : int
        maximum number of iterations to be done

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped data, with all "no-data value"
        pixels removed from its outer edges. The data type of the array is preserved.

    Raises
    ------
    ValueError
        If the `depth` parameter is `None` or an empty array (e.g., has a zero dimension).
    TypeError
        If `depth` is not a `numpy.ndarray`.

    Notes
    -----
    This iterative approach can be computationally expensive for very large
    arrays or arrays with thick borders of no-data values, as it creates
    new array views in each iteration. For performance-critical applications
    on massive datasets, a more vectorized approach to find the first/last
    valid rows/columns might be considered, though this iterative method
    is generally robust for typical TIFF edge cases.

    The function includes a safeguard (100 iterations) to prevent infinite
    loops on peculiar data, but assumes that inner elements are generally valid.

    Examples
    --------
    >>> import numpy as np
    >>> # Example 1: Basic cropping
    >>> data = np.array([
    ...     [99, 99, 99, 99],
    ...     [99,  1,  2, 99],
    ...     [99,  3,  4, 99],
    ...     [99, 99, 99, 99]
    ... ])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 2: No cropping needed
    >>> data = np.array([[1, 2], [3, 4]])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 3: Different NDV (e.g., np.nan)
    >>> data_nan = np.array([
    ...     [np.nan, np.nan, 1.0, np.nan],
    ...     [np.nan, 2.0, 3.0, np.nan],
    ...     [np.nan, np.nan, np.nan, np.nan]
    ... ])
    >>> cropped_data_nan = remove_edge_Nans(data_nan, np.nan)
    >>> print(cropped_data_nan)
    [[1. 2.]
     [3. 4.]]

        """

    # Type check for depth
    if not isinstance(depth, np.ndarray):
        raise TypeError("Input 'depth' must be a NumPy array (np.ndarray).")
    
    # Handle initial empty array or None input
    if depth is None or depth.size == 0:
        raise ValueError("Input 'depth' array cannot be None or empty.")

    
    # Create a working copy to avoid modifying the original array passed in
    elev = depth.copy() 
    original_shape = depth.shape
    
    # Set up value for max_iteration if none declared
    if max_iterations == None:
        max_iterations = int(original_shape[0]/2) 
        if verbose:
            print("Setting max iteration to half the size of depth array")
    
    if np.isnan(ndv):
        def is_ndv(data_array):
            return np.any(np.isnan(data_array))
    else:
        def is_ndv(data_array):
            return np.any(data_array == ndv)

    shrink_idx = 0
    have_ndv = True
    # remove edges that have NaN elements
    # continue until all edges are NaN free or exceeded 100 iterations
    # assumes that all inner elements are non NaN
    while have_ndv:
        tmp = elev[0,:]
        if is_ndv(tmp):
            elev = elev[1:,:]
        tmp = elev[:,0]
        if is_ndv(tmp):
            elev = elev[:,1:]
        tmp = elev[-1,:]
        if is_ndv(tmp):
            elev = elev[:-1,:]
        tmp = elev[:,-1]
        if is_ndv(tmp):
            elev = elev[:,:-1]
        shrink_idx +=1
        have_ndv = is_ndv(elev)
        if shrink_idx > max_iterations:
            have_ndv = False
            
    # if np.any(elev == ndv)
            
    if verbose:
        print(f"""
              Total of {shrink_idx} outer layers removed.
              No data value: {ndv}
              Original data size: {original_shape}
              New data size: {elev.shape}
            """)
        if shrink_idx >= max_iterations:
            print(f"Warning: Cropping stopped because max_iterations ({max_iterations}) was reached.")
            if np.any(elev == ndv):
                print(f"CRITICAL: Return data still contains NDV values.")
            
    return elev