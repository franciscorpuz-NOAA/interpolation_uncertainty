import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def get_column_indices(
    array_len: int,
    resolution: int,
    linespacing_meters: int,
    max_multiple: int,
    verbose: bool = False,
):
    """
    Determine indices of the columns to be used as sampling lines
    Dependent on the maximum multiple of the linespacing to be used as window size

    Parameters:
    -----------
    array_len : int
                Length of the array containing bathymetry data
    resolution : float
                    Spatial Resolution of the bathymetry data
    linespacing_meters : int
                            distance between 2 vertical sample lines (m)
    max_multiple : int
                    maximum multiple of the linespacing to be used as window size for FFT


    Returns:
    -----------
    col_indxs : np.array
                array indices corresponding to sample points/lines

    """

    # Round the desired linespacing to the nearest even integer for computational convenience
    linespacing_in_pixels = np.round(linespacing_meters / resolution)
    if (linespacing_in_pixels % 2) != 0:
        linespacing_in_pixels = linespacing_in_pixels - 1

    if array_len < linespacing_in_pixels:
        raise ValueError(
            f"""Data length should be greater than the desired linespacing.
                            Entered Linespacing: {linespacing_meters}m
                            Survey width: {array_len * resolution}m"""
        )

    # The first and last columns to be sampled is determined by the window size/length
    window_size_pixels = linespacing_in_pixels * max_multiple
    start_col = int(
        (window_size_pixels // max_multiple) - (linespacing_in_pixels // max_multiple)
    )
    last_col = int(
        array_len
        - (window_size_pixels // max_multiple)
        + (linespacing_in_pixels // max_multiple)
        - 1
    )

    # actual sampling indices will be determined by the desired linespacing
    col_indxs = np.arange(start_col, last_col, (linespacing_in_pixels + 1)).astype(int)

    if verbose:
        print(f"File width: {array_len}")
        print(f"Max multiple for window size: {max_multiple}")
        print(f"Desired linespacing (m): {linespacing_meters}")
        print(
            f"Linespacing (m)/(pixel) to be used: {int(linespacing_in_pixels * resolution)}m/{int(linespacing_in_pixels)}pixels"
        )

    return col_indxs


def get_strip(
    depth: np.array,
    column_indices: tuple[int, int],
    multiple: int,
    verbose: bool = False,
) -> np.array:
    """
    Retrieve section of the bathymetric data for FFT processing given
    column indices and multiple of the linespacing

    This function computes the strip of data for FFT processing. The width
    of the strip is the effective window size. The window size is the product
    of the linespacing and the value of the multiple parameter.

    Parameters
    ----------
    depth : np.array
            Bathymetric/depth data
    column_indices : np.array
                    column indices/location of the sampling lines
    multiple : int
            Determines the width of the window size with reference to the linespacing

    Returns
    --------
    segment : np.array
            a segment of the bathymetric data for further FFT processing


    """

    current_multiple = multiple
    start, end = column_indices[0], column_indices[1]
    linespacing = end - start + 1
    window_size = linespacing * current_multiple
    midpoint = start + (linespacing // 2) + 1

    # Determine indices for the window segment, include sampling columns
    start_col = int(midpoint - (window_size // 2))
    end_col = int(midpoint + (window_size // 2))
    window = depth[:, start_col:end_col]

    if verbose:
        print(
            f"""
            Current column: {start}
            Current multiple: {current_multiple}
            Midpoint: {midpoint}
            Linespacing: {linespacing}
            start_col: {start_col}
            end_col: {end_col}
            window length: {len(window)}
        """
        )
        plt.figure()
        plt.plot(np.arange(start_col, end_col), window, label="Window")
        locs, _ = plt.xticks()
        plt.xticks(locs, labels=[int(x + start_col) for x in locs], rotation=90)
        plt.vlines(
            [start_col, end_col, midpoint],
            ymin=np.min(window),
            ymax=np.max(window),
            color="gray",
            linestyle="--",
            alpha=0.2,
        )
        plt.xlabel("Data Columns (pixels)")
        plt.ylabel("Depth (m)")
        plt.legend()

    return window


def compute_residual(data_strip: np.array, verbose: bool = False) -> np.array:
    """
    Compute the residual error from estimating the data using linear interpolation

    This function computes the estimate for the data strip using the edge values
    and returns the residual error

    Parameters
    ----------

    data_strip : np.array
                 Raw bathymetric data
    verbose : bool
              Flag for printing useful details for debugging


    Returns
    -------
    residual : np.array
               Difference of the interpolation from the input data strip

    """
    interpolated_strip = np.array(
        [np.linspace(start=x[0], stop=x[-1], num=len(x)) for x in data_strip]
    )
    residual = data_strip - interpolated_strip

    if verbose:
        print(f"Input length: {len(data_strip)}")
        print(f"Interpolation length: {len(interpolated_strip)}")

        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(data_strip, label="Depth")
        plt.plot(interpolated_strip, label="Interpolation")
        plt.vlines(
            [0, len(residual)],
            ymin=np.min(data_strip),
            ymax=np.max(data_strip),
            color="gray",
            linestyle="--",
            alpha=0.2,
        )
        plt.xlabel("Data Columns (pixels)")
        plt.ylabel("Bathymetry (m)")
        plt.legend()
        plt.title(f"Interpolated Data")

        plt.subplot(122)
        plt.plot(residual, label="Residual")
        plt.hlines(
            [0], xmin=0, xmax=len(residual), color="gray", linestyle="--", alpha=0.2
        )
        plt.xlabel("Data Columns (pixels)")
        plt.ylabel("Residual Error (m)")
        plt.legend()
        plt.title(f"Residual Data")

    return residual, interpolated_strip


def preprocess_fft_input(data: np.array, windowing: str, verbose: bool = False):
    """
    Apply pre-processing to FFT input

    Parameters
    ----------
    data : np.array
           Input signal for FFT extraction
    windowing : str
                Type of window to be applied to input
    verbose : bool
              Flag for printing useful details for debugging


    Returns
    --------
    output : np.array
             Processed signal
    scale_factor : float
                   Scale factor to recover lost energy due to windowing

    """

    preprocessed_data = data - np.mean(data)
    # preprocessed_data = data
    segment_window = signal.windows.get_window(
        window=windowing, Nx=len(data), fftbins=False
    )
    preprocessed_data = preprocessed_data * segment_window
    # scale_factor = len(segment_window)/np.sum(segment_window)
    scale_factor = np.sum(segment_window)

    if verbose:
        print("Function: preprocess_fft_input")
        print(f"len input: {len(data)}")
        print(f"len window: {len(segment_window)}")
        print(f"len output: {len(preprocessed_data)}")

    return preprocessed_data, scale_factor


def compute_energy(
    data: np.array,
    resolution: int,
    method: str,
    window_values: np.array,
    verbose: bool = False,
) -> np.array:
    """
    Compute FFT energy using 'method' process

    Parameters
    ----------
    data : np.array
           Input data
    resolution : int
                 Spatial resoluton of the array
    method : str
             FFT Method used to estimate signal energy
    verbose : bool
              Flag for printing useful details for debugging

    Returns
    -------
    np.array
            Spectral energy in the signal
    """

    rfft_values = np.abs(np.fft.rfft(data))
    r_frequencies = np.fft.rfftfreq(len(data), d=resolution)
    if method == "amplitude":
        scale_factor = np.sum(window_values)
        if len(rfft_values) % 2 == 0:
            rfft_values[1:-1] = rfft_values[1:-1] * 2
            rfft_values = rfft_values[:-1]
            r_frequencies = r_frequencies[:-1]
        else:
            rfft_values[1:] = rfft_values[1:] * 2

    elif method == "psd":
        scale_factor = np.sum(window_values**2) * (1 / resolution)
        rfft_values = rfft_values**2
        if len(rfft_values) % 2 == 0:
            rfft_values[1:-1] = rfft_values[1:-1] * 2
            rfft_values = rfft_values[:-1]
            r_frequencies = r_frequencies[:-1]
        else:
            rfft_values[1:] = rfft_values[1:] * 2

    elif method == "spectrum":
        scale_factor = np.sqrt(np.sum(window_values) ** 2) * (1 / resolution)
        rfft_values = rfft_values**2
        if len(rfft_values) % 2 == 0:
            rfft_values[1:-1] = rfft_values[1:-1] * 2
            rfft_values = rfft_values[:-1]
            r_frequencies = r_frequencies[:-1]
        else:
            rfft_values[1:] = rfft_values[1:] * 2
    else:
        raise ValueError(
            f"""Unknown FFT Method: {method}
                         FFT options: {'amplitude', 'psd', 'spectrum'}
                         """
        )

    energy = rfft_values / scale_factor

    # fft_values = np.abs(np.fft.fft(normalized_input))
    # frequencies = np.fft.fftfreq(len(normalized_input))

    if verbose:
        print(
            f"""
              Function: compute_energy
              Length of input: {len(data)}
              Length of output energy: {len(energy)}
              Length of output frequencies: {len(r_frequencies)}
              """
        )
        plt.figure()
        plt.stem(r_frequencies, energy, label=f"{method}")
        plt.title(f"Window PSD, sum:{np.sum(energy):.02f}")
        plt.ylabel("FFT Energy")
        plt.legend()
        plt.xlabel("Frequencies")

    return energy, r_frequencies


def create_spatial_signal(resolution: int, max_cell_number: int):
    """
    Create the distance and frequency dependent scaling factors.
    """
    frequencies = np.fft.rfftfreq(max_cell_number, resolution)
    if len(frequencies) % 2 == 0:
        frequencies = frequencies[:-1]
    distances = np.arange(max_cell_number) * resolution
    distances_2d, freq_2d = np.meshgrid(distances, frequencies)
    spatial_scale = distances_2d * freq_2d
    spatial_scale = np.where(spatial_scale < 0.25, spatial_scale, 0.25)
    spatial_signal = np.sin(spatial_scale * 2 * np.pi)

    return spatial_signal


def compute_fft_uncertainty(
    data: np.array,
    multiple: int,
    resolution: int,
    windowing: str = None,
    method: str = "amplitude",
    selection: str = "half",
    verbose: bool = False,
) -> np.array:
    """
    Estimate the uncertainty using FFT

    Parameters
    ----------
    data : np.array
           Input data for FFT estimation
    multiple : int
               Window length as multiple of the linespacing
    resolution : int
                 Input data resolution for frequency calculation
    windowing : str
                Type of window to taper input
                options: scipy.signal.windows type
    method : str
             Type of FFT to estimate energy, defaults to 'amplitude'
             options: ['amplitude', 'density_trapezoid', 'density_sum', 'spectrum]
    verbose : bool, optional


    Returns
    -------
    np.array
        Uncertainty estimate from the FFT method
        To be compared with the residual error

    """

    # preprocessed_signal, scale_factor = preprocess_fft_input(data, windowing, verbose=False)
    preprocessed_data = data - np.mean(data)
    segment_window = signal.windows.get_window(
        window=windowing, Nx=len(data), fftbins=False
    )
    preprocessed_signal = preprocessed_data * segment_window
    psd, psd_freqs = compute_energy(
        preprocessed_signal, resolution, method, segment_window, verbose=False
    )
    spatial_signal = create_spatial_signal(resolution, len(data))
    variance = psd @ spatial_signal
    if method == "amplitude":
        window_uncertainty = variance
    else:
        window_uncertainty = np.sqrt(variance) * (1 / resolution)

    # linespacing_width = int((len(data)) // multiple)
    linespacing_width = int(len(data))
    estimate = np.zeros(linespacing_width)  # Include sampling lines in output length

    if selection == "half":
        estimate[: (linespacing_width // 2)] = window_uncertainty[
            : (linespacing_width // 2)
        ]
        estimate[-(linespacing_width // 2) :] = np.flip(
            window_uncertainty[: (linespacing_width // 2)]
        )
    else:
        freqs_window = np.fft.rfftfreq(int(len(data) / multiple), resolution)
        freq_idxs = np.where(np.isin(freqs_window, psd_freqs))[0]
        estimate[: (linespacing_width // 2)] = window_uncertainty[freq_idxs]
        estimate[(linespacing_width // 2) :] = np.flip(window_uncertainty[freq_idxs])

    if verbose:

        print(
            f"""
              Window length: {len(data)} 
              Linespacing width: {linespacing_width}
              Estimate length: {len(estimate)}
              """
        )
        plt.figure()
        plt.plot(data, label=f"Input (max:{np.max(data):.02f})")
        plt.plot(np.abs(data), label=f"Input, Abs (max:{np.max(data):.02f})")
        # plt.plot(preprocessed_signal, label="Preprocessed")
        plt.plot(estimate, label=f"FFT Uncertainty (max:{np.max(estimate):.04f})")
        plt.ylabel("Depth values (m)")
        plt.title(f"Residual vs Uncertainty, Window: {windowing}")
        plt.legend()

    return estimate
