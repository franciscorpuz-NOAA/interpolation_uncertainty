import numpy as np
import matplotlib.pyplot as plt


def get_interpolation_residual(data, verbose=False):
    """
    Calculate the residual between the interpolated data and the original data.
    Interpolated data is a line between the two end values.
    """
    # data_shape = data.shape
    start, end = 0, len(data)

    slopes = (data[-1] - data[0]) / len(data)
    start_vals = data[0]
    interpolation = np.arange(len(data))
    interpolation = interpolation * slopes + start_vals
    residual = data - interpolation

    if verbose:
        print(f"start: {start}")
        print(f"end: {end}")
        print(f"slope: {slopes}")
        print(f"interpolation: {interpolation}")

    return residual, interpolation


def preprocess_strip(data):
    """
    Remove the mean and apply a Hann window to the rows of the provided data array.
    """
    row_means = np.mean(data)
    data = data - row_means
    shape = len(data)
    window = np.hanning(shape)
    data = data * window
    return data


def psd_strip_modified(data, resolution):
    """
    Calculate the power spectral density of the provided data array.

    For background on the inline comments, see https://docs.scipy.org/doc/scipy/tutorial/signal.html#sampled-sine-signal
    """
    shape = len(data)
    hann_win = np.hanning(shape)
    hann_scale = 1 / np.sqrt(np.sum(np.square(hann_win)))
    fft_result = np.fft.rfft(data) * hann_scale  # this is $X_l^w$
    psd_result = 2 * resolution * np.square(np.abs(fft_result))
    # drop zero frequency
    frequencies = np.fft.rfftfreq(shape, resolution)[1:]
    psd_result = psd_result[1:shape]
    return psd_result, frequencies


def amp_strip(data, resolution):
    """
    Calculate the power spectral density of the provided data array.

    For background on the inline comments, see https://docs.scipy.org/doc/scipy/tutorial/signal.html#sampled-sine-signal
    """
    # shape = data.shape
    data = preprocess_strip(data)
    fft_result = np.fft.rfft(data)
    amp_result = 2 * np.abs(fft_result)
    # drop zero frequency
    frequencies = np.fft.rfftfreq(len(data), resolution)[1:]
    amp_result = amp_result[1:]
    return amp_result, frequencies


class spatial_contributions:
    def __init__(self, resolution, max_cell_number):
        """
        Create the distance and frequency dependent scaling factors.
        """
        # self.frequencies = frequencies inside the window
        self.frequencies = np.fft.rfftfreq(max_cell_number, resolution)[1:]
        self.distances = np.arange(max_cell_number) * resolution
        distances_2d, freq_2d = np.meshgrid(self.distances, self.frequencies)
        self.spatial_scale = distances_2d * freq_2d
        self.spatial_scale = np.where(
            self.spatial_scale < 0.25, self.spatial_scale, 0.25
        )
        self.spatial_signal = np.sin(self.spatial_scale * 2 * np.pi)

    def get_uncertainties(self, pxx, fft_freq, interpolation_cell_distance):
        """
        Multiply the scaling factors by the amplitude to get the uncertainties.
        """
        freq_idx = np.where(np.isin(fft_freq, self.frequencies))[0]
        # freq_contributions = psd * self.spatial_signal[freq_idx,:]
        # should add a check to make sure self.frequencies are matched appropriately to psd_freq
        window_uncertainties = pxx @ self.spatial_signal / len(freq_idx)  # sum and *
        # print(f"normalization glen: {len(freq_idx)}")
        # print(f"len fft_freq: {len(fft_freq)}")
        # print(f"len self freqs: {len(self.frequencies)}")
        interpolation_cell_distance = int(interpolation_cell_distance)
        interpolation_uncertainties = np.zeros((interpolation_cell_distance))
        interpolation_uncertainties[: (interpolation_cell_distance // 2)] = (
            window_uncertainties[: (interpolation_cell_distance // 2)]
        )
        interpolation_uncertainties[(interpolation_cell_distance // 2) :] = np.flip(
            window_uncertainties[: (interpolation_cell_distance // 2)]
        )

        return interpolation_uncertainties


def glen_get_uncertainties(data, resolution, multiple, method="amplitude"):
    data = preprocess_strip(data)
    if method == "amplitude":
        energy, freq = amp_strip(data, resolution)
        scaler = spatial_contributions(resolution, len(data))
        uncertainty = scaler.get_uncertainties(
            energy, freq, round(len(data) / multiple)
        )
    elif method == "psd":
        energy, freq = psd_strip_modified(data, resolution)
        scaler = spatial_contributions(resolution, len(data))
        uncertainty = scaler.get_uncertainties(
            energy, freq, round(len(data) / multiple)
        )
        uncertainty = np.sqrt(uncertainty)
    else:
        raise ValueError(f"Unrecognized Method: {method}")

    return uncertainty


def uncertainty_comparison(residuals, uncertainties, plot=False):
    nonzero_idx = np.nonzero(
        (residuals != 0) & (~np.isnan(residuals)) & (uncertainties != 0)
    )
    uncertainty_ratio = np.full(residuals.shape, np.nan)
    uncertainty_ratio[nonzero_idx] = uncertainties[nonzero_idx] / np.abs(
        residuals[nonzero_idx]
    )
    fail_points = np.nonzero(uncertainty_ratio < 1)
    ur_flat = uncertainty_ratio[nonzero_idx].flatten()
    total_count = len(ur_flat)
    fail_count = len(fail_points[0])
    pass_percentage = 100 - fail_count / total_count * 100
    current_rmse = np.sqrt(
        np.mean((residuals[nonzero_idx] - uncertainties[nonzero_idx]) ** 2)
    )
    mean_error = np.mean(
        np.mean(np.sum(uncertainties[nonzero_idx]) - np.abs(residuals[nonzero_idx]))
        # normalize with number of points
        # review this one
    )
    if plot:
        print(f"mean difference: {mean_error}")
        plt.figure()
        plt.hist(ur_flat, bins=100, log=True)
        plt.xlim(0, 5)
        plt.xlabel("Uncertainty Ratio")
        plt.ylabel("Count")
        plt.grid()
        plt.show()
        max_uncertainty = np.max(uncertainties[nonzero_idx])
        plt.figure()
        plt.plot(np.abs(residuals.T), uncertainties.T, ".")
        plt.plot([0, max_uncertainty], [0, max_uncertainty], "r")
        plt.grid()
        plt.xlabel("Residual")
        plt.ylabel("Uncertainty Estimate")
        plt.title(
            f"""Pass Percentage: {pass_percentage:.2f}%, Total Point Count: {total_count}, Failed Point Count: {fail_count}
                RMSE: {current_rmse:.02f}, Mean Difference: {mean_error:.02f}"""
        )
        plt.show()

    return {
        "fail_pts": fail_count,
        "total_pts": total_count,
        "percentage": pass_percentage,
        "mean": mean_error,
        "rmse": current_rmse,
    }
