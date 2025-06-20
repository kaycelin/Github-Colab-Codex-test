import numpy as np


def time_delayer(data, delay_time_ps, freq, samp_rate, do_phase_delay=True, do_time_delay=False):
    """Apply phase and optional time delay to a signal."""
    delayed_data = data
    if do_phase_delay:
        delayed_data = delayed_data * np.exp(1j * 2 * np.pi * freq * delay_time_ps * 1e-12)
    if do_time_delay:
        delay = samp_rate * delay_time_ps * 1e-12
        N = 21
        n = np.arange(-N//2, N//2)
        h = np.sinc(n - delay)
        h *= np.hamming(N)
        h /= np.sum(h)
        delayed_data = np.convolve(delayed_data, h, mode='same')
    return delayed_data


def dbfs(raw_data):
    """Return FFT magnitude in dBFS."""
    win = np.hamming(len(raw_data))
    s_fft = np.fft.fft(raw_data * win) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20 * np.log10(np.abs(s_shift) / (2**11))
    return s_dbfs


def calcTheta(delay_ps, c=3e8, d=0.5 * (3e8 / 2.3e9)):
    """Convert path delay (ps) to angle in degrees."""
    arcsin_arg = delay_ps * 1e-12 * c / d
    arcsin_arg = np.clip(arcsin_arg, -1, 1)
    return np.rad2deg(np.arcsin(arcsin_arg))
