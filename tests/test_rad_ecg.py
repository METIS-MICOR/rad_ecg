import pytest
import numpy as np
from pathlib import Path

# Adjust imports based on your actual module names
from peak_detect_v4 import HeartBeat, ECGData, CardiacFreqTools, RadECG

# ==========================================
# FIXTURES (Mock Data & Objects)
# ==========================================

@pytest.fixture
def fs():
    return 1000.0

@pytest.fixture
def freq_tools(fs):
    return CardiacFreqTools(fs=fs)

@pytest.fixture
def mock_rad_ecg(fs):
    """Provides a lightweight RadECG object for testing internal methods."""
    wave = np.zeros(int(fs))
    data = ECGData(fs=fs, wave=wave)
    data.sect_info = np.zeros(1) 
    configs = {"plot_section": False, "plot_errors": False}
    fp = Path("dummy_path")
    return RadECG(data, configs, fp)

@pytest.fixture
def signal_10hz(fs):
    """Perfect QRS-band signal (10Hz sine wave)."""
    t = np.arange(int(fs * 2)) / fs
    return np.sin(2 * np.pi * 10 * t)

@pytest.fixture
def signal_30hz(fs):
    """High-frequency muscle artifact signal (30Hz sine wave)."""
    t = np.arange(int(fs * 2)) / fs
    return np.sin(2 * np.pi * 30 * t)

@pytest.fixture
def signal_noise(fs):
    """Pure broadband white noise."""
    np.random.seed(42)
    return np.random.normal(0, 1, int(fs * 2))

@pytest.fixture
def signal_flatline(fs):
    """Dead sensor with micro-voltage static."""
    np.random.seed(42)
    return np.random.normal(0, 0.0001, int(fs * 2))

# ==========================================
# DATA STRUCTURE TESTS
# ==========================================

def test_heartbeat_to_row_null_safety():
    """Ensures un-extracted peaks (None) are safely cast to 0 for Numpy structured arrays."""
    beat = HeartBeat(r_peak=500)
    # R-peak is populated, everything else is None default
    row = beat.to_row()
    
    assert row[2] == 500  # r_peak is index 2
    assert row[0] == 0    # p_peak (None) becomes 0
    assert None not in row # No Nones should escape

# ==========================================
# SIGNAL QUALITY (SQI) MATH TESTS
# ==========================================

def test_hjorth_complexity(freq_tools, signal_10hz, signal_noise):
    """Hjorth complexity should be distinctly higher for broadband noise than a pure sine wave."""
    hjorth_sine = freq_tools.calc_hjorth_complexity(signal_10hz)
    hjorth_noise = freq_tools.calc_hjorth_complexity(signal_noise)
    
    assert hjorth_noise > hjorth_sine
    assert hjorth_sine < 1.5  # Pure sine waves have very low complexity

def test_spectral_power_ratios(freq_tools, signal_10hz, signal_30hz):
    """Tests if the Welch PSD correctly isolates the 5-15Hz QRS band."""
    qrs_ratio_10hz, _, _, _ = freq_tools.calc_spec_metrics(signal_10hz)
    qrs_ratio_30hz, _, _, _ = freq_tools.calc_spec_metrics(signal_30hz)
    
    # 10Hz signal should have almost all its power in the QRS ratio
    assert qrs_ratio_10hz > 0.90
    # 30Hz signal should have almost zero power in the QRS ratio
    assert qrs_ratio_30hz < 0.10

def test_spectral_entropy(freq_tools, signal_10hz, signal_noise):
    """Entropy should be low for a single frequency, and high for scattered noise."""
    _, _, _, ent_10hz = freq_tools.calc_spec_metrics(signal_10hz)
    _, _, _, ent_noise = freq_tools.calc_spec_metrics(signal_noise)
    
    # threshold to < 2.0 to account for Welch spectral leakage
    assert ent_10hz < 2.0  
    assert ent_noise > 5.0 # Highly chaotic

def test_pre_peak_sqi_flatline(freq_tools, signal_flatline):
    """A dead sensor should instantly trip the Kurtosis logic gate."""
    is_valid, fail_reason, metrics = freq_tools.pre_peak_sqi(signal_flatline)
    
    assert not is_valid
    assert "Low Kurtosis" in fail_reason

# ==========================================
# 3. ECG INTERVAL FORMULA TESTS
# ==========================================

def test_calc_qtc_bazett(mock_rad_ecg):
    """QTc Bazett = QT / sqrt(RR in seconds). QT=400ms, RR=1000ms -> QTc=400ms."""
    qtc = mock_rad_ecg._calc_qtc(QT=400, RR=1000, formula="Bazzett")
    assert qtc == 400

    # At 120bpm (RR=500ms), QTc should stretch significantly higher than QT
    qtc_fast = mock_rad_ecg._calc_qtc(QT=300, RR=500, formula="Bazzett")
    assert qtc_fast == 424 # 300 / sqrt(0.5) = 424.26

def test_calc_qtc_zero_division(mock_rad_ecg):
    """Ensures formula doesn't crash on missing/zero RR intervals."""
    qtc = mock_rad_ecg._calc_qtc(QT=400, RR=0, formula="Bazzett")
    assert qtc == 0

def test_calc_qtvi(mock_rad_ecg):
    """Tests the Berger QTVI logarithmic equation."""
    qt_list = [400, 405, 395, 410]
    rr_list = [1000, 950, 1050, 900]
    
    qtvi = mock_rad_ecg._calc_qtvi(qt_list, rr_list)
    assert isinstance(qtvi, float)
    assert not np.isnan(qtvi)

def test_calc_qtvi_nan_handling(mock_rad_ecg):
    """QTVI requires at least 2 valid beats to calculate variance."""
    qt_list = [400, None, np.nan]
    rr_list = [1000, 1000, 1000]
    
    qtvi = mock_rad_ecg._calc_qtvi(qt_list, rr_list)
    assert np.isnan(qtvi)

# ==========================================
# HISTORICAL LOGIC TESTS
# ==========================================

def test_consecutive_valid_peaks(mock_rad_ecg):
    """Tests if the pipeline successfully locates a continuous block of historical valid peaks."""
    # Create a mock peaks array: [Index, ValidFlag]
    # Beats 0-4 are valid. Beat 5 is invalid. Beats 6-10 are valid.
    mock_peaks = np.array([
        [1000, 1],
        [2000, 1],
        [3000, 1],
        [4000, 1],
        [5000, 1],
        [6000, 0], # <-- Gap in history!
        [7000, 1],
        [8000, 1],
        [9000, 1],
    ])
    
    # Looking back 1999 samples. It will stop at index 7000 because 
    # 9000 - 7000 = 2000, which is strictly > 1999.
    result = mock_rad_ecg.consecutive_valid_peaks(mock_peaks, lookback=1999)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert result[0] == 7000
    assert result[-1] == 9000
