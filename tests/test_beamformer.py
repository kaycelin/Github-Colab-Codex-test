import numpy as np
import beamformer


def test_filter_length(monkeypatch):
    called = {}
    orig = np.convolve

    def fake_convolve(data, h, mode="full"):
        called['length'] = len(h)
        return orig(data, h, mode=mode)

    monkeypatch.setattr(np, 'convolve', fake_convolve)
    data = np.ones(10)
    beamformer.time_delayer(data, 100, 1e6, 1e6, do_phase_delay=False, do_time_delay=True)
    assert called['length'] == 21


def test_zero_delay_identity():
    data = np.random.randn(50) + 1j * np.random.randn(50)
    out = beamformer.time_delayer(data, 0, 1e6, 1e6, do_phase_delay=True, do_time_delay=True)
    np.testing.assert_allclose(out, data, atol=1e-6)


def test_calcTheta():
    theta = 30.0
    c = 3e8
    d = 0.5 * (c / 2.3e9)
    delay_ps = d * np.sin(np.deg2rad(theta)) / c * 1e12
    assert np.isclose(beamformer.calcTheta(delay_ps, c=c, d=d), theta, atol=1e-6)
