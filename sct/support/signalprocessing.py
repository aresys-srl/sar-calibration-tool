# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import resample


def fft_fitting(vect, weights, vect_mean, nyquist_position):

    n = len(vect)

    # Linear fitting
    vect_fit = weights * np.exp(1j * 2 * np.pi * vect)  # weighting applied to estimates
    vect_fit_fft = np.fft.fftn(vect_fit, s=(n * 16,), axes=(0,))  # FFT interpolation factor = 16
    pos = np.argmax(np.abs(vect_fit_fft))

    if pos > n * 16 / 2 - 1:
        pos = pos - n * 16

    if pos == 0:
        vect = vect_mean * np.ones((1, n))
    else:
        vect_fit = np.arange(n) * (pos / 16 / n)

        # Ensure circularity (first sample = last sample)
        vect = vect_fit + vect[0]  # align first sample
        vect[nyquist_position + 1 :] = vect[nyquist_position + 1 :] - vect[-1] + vect[0]

    return vect


def modulate_data(data_portion, modulation_frequency):

    # Retrive the size of the input data
    ny, nx = data_portion.shape

    # Matrix of modulation
    # Note that every row of the modulation_matrix are equal and just varies along column to perform modulation along this direction
    if modulation_frequency.size == 1:
        modulation_frequency = modulation_frequency * np.ones((1, nx))

    arg = (2 * np.pi * np.arange(ny).reshape(-1, 1)) * modulation_frequency
    modulation_matrix = np.exp(1j * arg)

    # Performs modulation
    data_portion_modulated = data_portion * modulation_matrix

    return data_portion_modulated


def get_local_frequency(data_portion, method="AutoCorrelation"):

    n_rg, n_az = data_portion.shape

    if method == "AutoCorrelation":

        temp_corr = np.zeros((3, n_az), dtype=np.complex_)
        for ii in range(n_az):
            temp_corr[:, ii] = np.correlate(data_portion[:, ii], data_portion[:, ii], mode="full")[n_rg - 2 : n_rg + 1]
        tot_corr = np.sum(temp_corr, axis=1)

        local_frequency = np.angle(tot_corr[2]) / 2 / np.pi
        local_frequency_vect = np.angle(temp_corr[2, :]) / 2 / np.pi

    elif method == "FFT":
        raise NotImplementedError  # TODO
        """
        # #data_portion=cast(data_portion,'single');
        # data_portion_fft=np.fft.fft(data_portion, axis=0)
        # tempfft=np.fft.ifft(data_portion_fft * data_portion_fft.conjugate(), axis=0)
        # totfft=np.sum(tempfft,1)/n_az
        #
        # local_frequency=np.angle(totfft[1])/2/np.pi
        # local_frequency_vect=np.angle(tempfft[1,:])/2/np.pi
        """

    elif method == "PowerBalance":
        raise NotImplementedError  # TODO
        """
        # ovsFact = 1
        # n = 2 * pow235(np.ceil(n_rg/2.) * ovsFact, 1) # even
        # hn = n/2
        # data_portion_fft = np.fft.fft(data_portion, n, axis=0)
        # S = np.abs(data_portion_fft * data_portion_fft.conjugate())
        #
        # cs = np.cumsum(S)
        # P = np.ones((n,1)) * cs[-1,:]
        # cs = cs / P
        #
        # hp = cs[hn:-1,:] - cs[0:hn,:]
        # idx = np.argmin(np.abs(hp - 0.5))
        #
        # idx1 = idx
        # idx1[idx==hn] = idx[idx==hn] - 1
        # #local_frequency_vect = (idx-1)/n + 0.5*(hp(idx1+1) >= hp(idx1));
        # #local_frequency_vect = mod(idx/n + 0.5*(hp(idx1+1) >= hp(idx1)), 1);
        # local_frequency_vect = np.double(idx)/n
        # for ii in range(0, n_az):
        #     if hp[idx1[ii]+1,ii] >= hp[idx1[ii],ii]:
        #         local_frequency_vect[ii] = local_frequency_vect[ii] + 0.5
        #
        # local_frequency_vect = ((local_frequency_vect + 0.5) % 1) - 0.5
        # local_frequency = np.mean(local_frequency_vect)
        # # local_frequency = sum(local_frequency_vect.*P(end,:))./sum(P(end,:)); % TODO: check
        """

    else:
        raise ValueError("Method not supported.")

    return local_frequency, local_frequency_vect


def get_local_frequency2d(data_portion, method="AutoCorrelation"):

    # Retrive range dimension of input data
    n_rg, _ = data_portion.shape

    # Transpose input data to estimate first azimuth frequency
    data_portion = data_portion.transpose()

    # Apply FFT to other domain (allows to retrieve a more precise local_frequency_az_vect)
    data_portion_fft = np.fft.fftn(data_portion, axes=(1,))

    # Compute local frequencies
    local_frequency_az, local_frequency_az_vect = get_local_frequency(data_portion_fft, method)

    # Apply a linear fitting to the local frequencies (done to avoid unwrap errors)
    # Ensure that the local frequencies fitting is circular (first sample = last sample)
    pos = np.argmin(np.sum(np.abs(data_portion_fft), axis=0))  # find Nyquist position
    local_frequency_az_vect = fft_fitting(
        local_frequency_az_vect, np.abs(np.sum(data_portion_fft, axis=0)), local_frequency_az, pos
    )

    # Apply demodulation
    data_portion_fft = modulate_data(data_portion_fft, -local_frequency_az_vect)

    # On the demodulated data, compute the local frequencies in the other domain
    mat_temp = np.fft.ifftn(data_portion_fft, axes=(1,)).transpose()
    mat_temp_fft = np.fft.fftn(mat_temp, axes=(1,))
    local_frequency_rg, local_frequency_rg_vect = get_local_frequency(mat_temp_fft, method)
    pos = n_rg // 2 - 1
    local_frequency_rg_vect = fft_fitting(
        local_frequency_rg_vect, np.abs(np.sum(mat_temp_fft, axis=0)), local_frequency_rg, pos
    )

    return local_frequency_rg, local_frequency_rg_vect, local_frequency_az, local_frequency_az_vect


def interp1_modulated_data(data_portion, interpolation_factor, demodulation_flag, demodulation_frequency):

    # Retrive dimensions of input data to be interpolated
    n_rg, _ = data_portion.shape

    if demodulation_flag >= 1:
        # Apply FFT to other domain (to keep the same approach used for the local frequencies estimations in get_local_frequency2d)
        data_portion_fft = np.fft.fftn(data_portion, axes=(1,))

        # Apply demodulation
        data_portion_fft = modulate_data(data_portion_fft, -demodulation_frequency)
    else:
        data_portion_fft = data_portion

    # Apply zero padding in time domain (x2 factor)
    data_portion_fft = np.concatenate((data_portion_fft, np.zeros(data_portion_fft.shape)), axis=0)

    # Interpolate with FFT method
    data_portion_fft_int = resample(data_portion_fft, interpolation_factor * 2 * n_rg, axis=0)

    # Remove padded data
    data_portion_fft_int = data_portion_fft_int[0 : interpolation_factor * n_rg, :]

    if demodulation_flag >= 1:
        # Apply interpolation factor to local frequencies
        frequency_vect = demodulation_frequency / interpolation_factor

        # Apply re-modulation
        data_portion_fft_int = modulate_data(data_portion_fft_int, frequency_vect)

        # Apply IFFT to other domain (see first step)
        data_portion = np.fft.ifftn(data_portion_fft_int, axes=(1,))
    else:
        data_portion = data_portion_fft_int

    return data_portion


def interp2_modulated_data(data_portion, interpolation_factor_rg, interpolation_factor_az, do_rg, do_az):

    # Compute local frequencies
    if do_rg > 1 or do_az > 1:
        f_rg, f_rg_vect, f_az, f_az_vect = get_local_frequency2d(data_portion)
        pos = data_portion.shape[1] // 2 - 1
        f_rg_vect = np.concatenate(
            (
                f_rg_vect[0, 0 : pos + 1],
                f_rg_vect[0, pos]
                * np.ones(
                    (interpolation_factor_az - 1) * data_portion.shape[1],
                ),
                f_rg_vect[0, pos + 1 :],
            )
        )
    if do_az == 1:
        f_az, f_az_vect = get_local_frequency(data_portion.transpose())
        f_az_vect = f_az * np.ones((f_az_vect.shape[0],))
    if do_rg == 1:
        f_rg, f_rg_vect = get_local_frequency(data_portion)
        f_rg_vect = f_rg * np.ones((f_rg_vect.shape[0] * interpolation_factor_az,))

    # Perform interpolation along azimuth direction
    if interpolation_factor_az > 1:
        data_portion = data_portion.transpose()
        data_portion = interp1_modulated_data(data_portion, interpolation_factor_az, do_az, f_az_vect)
        data_portion = data_portion.transpose()

    # Perform interpolation along range direction
    if interpolation_factor_rg > 1:
        data_portion = interp1_modulated_data(data_portion, interpolation_factor_rg, do_rg, f_rg_vect)

    return data_portion


def parabolic_interpolation(vect):

    alpha = vect[0]
    beta = vect[1]  # max
    gamma = vect[2]
    delta_position = (np.abs(alpha) - np.abs(gamma)) / (np.abs(alpha) - 2 * np.abs(beta) + np.abs(gamma)) / 2
    peak_value = beta - (alpha - gamma) * delta_position / 4

    return peak_value, delta_position


def max2d(data_portion):

    ind = np.unravel_index(np.argmax(np.abs(data_portion), axis=None), data_portion.shape)

    return data_portion[ind], ind[0], ind[1]


def max2d_fine(data_portion, interpolation_factor=8, perform_demodulation=(1, 1)):

    # Implementation based on parabolic_interpolation

    # Size of the input matrix
    n_rg, n_az = data_portion.shape

    # First coarse peak estimation and range/azimuth cuts extraction
    _, y_max_pos_coarse, x_max_pos_coarse = max2d(data_portion)
    mat_x = data_portion[y_max_pos_coarse : y_max_pos_coarse + 1, :]
    mat_y = data_portion[:, x_max_pos_coarse : x_max_pos_coarse + 1]

    # Oversampling of each cut by a factor interpolation_factor, with centroid estimation and correction
    mat_x = interp2_modulated_data(mat_x, 1, interpolation_factor, perform_demodulation[0], perform_demodulation[1])
    mat_y = interp2_modulated_data(
        mat_y.transpose().conjugate(), 1, interpolation_factor, perform_demodulation[0], perform_demodulation[1]
    )

    # Coarse peak estimation of the oversampled signal and parabolic interpolation around maximum coordinates
    x_max_pos_coarse = np.argmax(np.abs(mat_x[0, :]))
    x_max_pos_coarse = np.min([np.max([1, x_max_pos_coarse]), mat_x.size - 1])
    _, x_delta_position = parabolic_interpolation(np.abs(mat_x[0, x_max_pos_coarse - 1 : x_max_pos_coarse + 2]))

    y_max_pos_coarse = np.argmax(np.abs(mat_y[0, :]))
    y_max_pos_coarse = np.min([np.max([1, y_max_pos_coarse]), mat_y.size - 1])
    _, y_delta_position = parabolic_interpolation(np.abs(mat_y[0, y_max_pos_coarse - 1 : y_max_pos_coarse + 2]))

    # Final peak position in [x y] coordinate and index correction.
    x_max_pos = x_delta_position + x_max_pos_coarse
    y_max_pos = y_delta_position + y_max_pos_coarse

    x_max_pos = x_max_pos / interpolation_factor
    y_max_pos = y_max_pos / interpolation_factor

    y_axis = np.arange(n_rg) - y_max_pos
    x_axis = np.arange(n_az) - x_max_pos

    filter_rg = np.sinc(y_axis)
    filter_az = np.sinc(x_axis)
    peak_value = np.matmul(filter_rg, np.matmul(data_portion, filter_az))

    return peak_value, y_max_pos, x_max_pos


def get_frequency_axis(central_frequency, sampling_frequency, n_samples):

    central_frequency = central_frequency.squeeze()

    frequency_shift = central_frequency % sampling_frequency

    sampling_step = sampling_frequency / n_samples

    starting_frequency = np.arange(0, n_samples) * sampling_step

    frequency_axis = np.zeros((central_frequency.size, n_samples))
    for t in range(central_frequency.size):
        f = (
            (starting_frequency - frequency_shift[t] + sampling_frequency / 2) % sampling_frequency
        ) - sampling_frequency / 2
        frequency_axis[t, :] = f + central_frequency[t]

    return frequency_axis


def shift_data(data_portion, shift_rg, shift_az):

    # HP: input shifts (shift_rg, shift_az) are considered expressed in samples and lines respectively
    # NOTE Data spectrum is supposed to be unfolded (e.g. for Topsar data, deramping already applied)

    if np.abs(shift_rg - np.round(shift_rg)) < 0.001 and np.abs(shift_az - np.round(shift_az)) < 0.001:
        # Perform shift adding zeros before/after the data
        # - Applying rounding to input shifts
        shift_rg = int(np.round(shift_rg))
        shift_az = int(np.round(shift_az))

        # - Derive input data sizes
        n_rg, n_az = data_portion.shape

        # - Apply range shift (if not 0)
        pad = np.zeros((np.abs(shift_rg), n_az))
        if shift_rg < 0:
            data_portion = np.concatenate((pad, data_portion[: -abs(shift_rg), :]), axis=0)
        elif shift_rg > 0:
            data_portion = np.concatenate((data_portion[abs(shift_rg) :, :], pad), axis=0)

        # - Apply azimuth shift (if not 0)
        pad = np.zeros((n_rg, abs(shift_az)))
        if shift_az < 0:
            data_portion = np.concatenate((pad, data_portion[:, : -abs(shift_az)]), axis=1)
        elif shift_az > 0:
            data_portion = np.concatenate((data_portion[:, abs(shift_az) :], pad), axis=1)

    else:
        # Perform shift applying to the data a phase ramp in frequency domain
        # - Derive input data sizes
        n_rg, n_az = data_portion.shape

        # - Derive range and azimuth frequencies
        f_rg, f_rg_vect, f_az, f_az_vect = get_local_frequency2d(data_portion)
        f_rg_vect = f_rg * np.ones(
            f_rg_vect.shape
        )  # NOTE Rg/az variance temporarily deactivated, to be made more robust
        f_az_vect = f_az * np.ones(f_az_vect.shape)

        # - Apply 2D FFT
        data_portion_fft = np.fft.fft2(data_portion)

        # - Compute and apply shifting kernel
        # TODO To be verified for TOPSAR data
        f_rg_axis = get_frequency_axis(f_rg_vect, 1, n_rg)
        f_az_axis = get_frequency_axis(f_az_vect, 1, n_az)
        phi = np.exp(1j * 2 * np.pi * (f_az_axis * shift_az + f_rg_axis.transpose() * shift_rg))
        data_portion_fft = data_portion_fft * phi

        # - Apply 2D IFFT
        data_portion = np.fft.ifft2(data_portion_fft)

    return data_portion
