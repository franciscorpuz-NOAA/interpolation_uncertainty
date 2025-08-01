from typing import Union, Any, Dict, Optional, Callable
from pathlib import Path
import numpy as np
import functools

from osgeo import gdal
gdal.UseExceptions()


def trace_function(func: Callable): 
    """
    Decorator used for debugging. Prints the input
    and corresponding outputs of the function "func"
    
    Parameters
    -----------
    func : Callable
           Function to inspect
     
    Returns
    --------
    """
    if not isinstance(func, Callable):
        raise TypeError(f"{func.__name__} is not a valid function.")
        
    @functools.wraps(func)   
    def wrapper(*args, **kwargs):
        original_result = func(*args, **kwargs)
        
        # print some diagnostics
        print(f"""
              Tracing: {func.__name__}():
              args: {args}
              kwargs: {kwargs}
              output: {original_result}
              output_len: {len(original_result)}""")
        
        # return function output
        return original_result
    
    return wrapper


def load_file(
    filename: Union[str, Path], folder: Optional[str] = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Reads bathymetry data from a TIFF file.

    This function uses the GDAL library to read a TIFF file
    containing bathymetric data. It extracts the primary data matrix,
    the "no-data" value (NDV), and the spatial resolution of the image.

    Parameters
    ----------
    filename : str, Path
        The base filename or Path of the TIFF file (e.g., "bathymetry.tif").
    folder : str, optional
        An optional path to the directory containing the TIFF file.
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
        If `filename` is not a string or Path.
        (Changed from ValueError for type-checking)
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

    if not isinstance(filename, (str, Path)):
        # Using TypeError for incorrect type, more conventional than ValueError
        raise TypeError("Filename must be a String or Path.")

    if folder:
        file_path = Path(folder) / filename
    else:
        file_path = Path(filename)

    # Check if the file exists before trying to open with GDAL
    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found at: {file_path}")

    with gdal.Open(str(file_path)) as ds:
        if not ds:
            raise RuntimeError(f"GDAL failed to open TIFF file: '{file_path}'")

        # Retrieve Bathymetric data
        depth_band = ds.GetRasterBand(1)
        if not depth_band:
            raise RuntimeError(f"Error retrieving depth data from {file_path}.")

        ndv = depth_band.GetNoDataValue()
        depth = depth_band.ReadAsArray()
        depth_gt = ds.GetGeoTransform()
        resolution = depth_gt[1]
        if resolution < 1:
            raise ValueError(
                f"Spatial resolution should be at least 1"
                f"Resolution value from file: {resolution}"
            )

    if verbose:
        # Print some statistics of the bathymetry data
        spatial_coverage_length = int(depth.shape[0] * resolution)
        spatial_coverage_width = int(depth.shape[1] * resolution)
        print(
            f"""Input filename: {str(file_path)}
              Data dimensions: {depth.shape}
              Min/Max: {np.nanmin(depth), np.nanmax(depth)}
              Survey Coverage: {spatial_coverage_length}m x {spatial_coverage_width}m
              Spatial Resolution: {resolution}
              """
        )

    # Compile bathymetric data in a dictionary and return directly
    return {
        "depth": depth,
        "ndv": ndv,
        "resolution": resolution,
        "dimensions": depth.shape,
    }


def remove_edge_Nans(
    depth: np.ndarray,
    ndv: Union[float, int, None, str],
    max_iterations: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Iteratively removes edge rows and columns containing no-data values.

    This function crops the input 2D array by iteratively removing outermost
    rows and columns containing "no-data" value (NDV).
    The process continues until all edges are free of NDV pixels
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
        Optional flag to print some information on the processed array
    max_iteration : int, optional
        maximum number of iterations to be done
        Default is half the largest dimension

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped data

    Raises
    ------
    ValueError
        If the `depth` parameter is `None` or an empty array
        (e.g., has a zero dimension).
    TypeError
        If `depth` is not a `numpy.ndarray`.

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
    if max_iterations is None:
        max_iterations = int(np.max([original_shape[0], original_shape[1]]) / 2)
        if verbose:
            print("Setting max iteration to half the size of depth array")

    if ndv == np.nan:
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
        tmp = elev[0, :]
        if is_ndv(tmp):
            elev = elev[1:, :]
        tmp = elev[:, 0]
        if is_ndv(tmp):
            elev = elev[:, 1:]
        tmp = elev[-1, :]
        if is_ndv(tmp):
            elev = elev[:-1, :]
        tmp = elev[:, -1]
        if is_ndv(tmp):
            elev = elev[:, :-1]
        shrink_idx += 1
        if not np.any(is_ndv(elev)):
            have_ndv = False
        if shrink_idx > max_iterations:
            break

    if verbose:
        print(
            f"""
              Total of {shrink_idx} outer layers removed.
              No data value: {ndv}
              Original data size: {original_shape}
              New data size: {elev.shape}
            """
        )
        if shrink_idx >= max_iterations:
            print(f"Warning: ({max_iterations}) max iterations was reached.")
        if np.any(elev == ndv):
            print("CRITICAL: Return data still contains NDV values.")

    return elev


def generate_augmentations(array: np.ndarray, num: int = 8):
    array_copy = array.copy()
    total_transformations = num
    for i in range(total_transformations):
        if i % 4 == 0:
            rotated_array = array_copy
        else:
            rotated_array = np.rot90(array_copy, k=i)
        if i // 4 > 0:
            rotated_array = np.fliplr(rotated_array)
        yield (i, rotated_array)


