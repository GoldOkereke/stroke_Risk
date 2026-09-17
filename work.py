import os
try:
    import wfdb
except ImportError:
    print("Module 'wfdb' is not installed. Install with: pip install wfdb")
    raise
import matplotlib.pyplot as plt
import numpy as np

# Record name (change to your local filename without extension)
record = '04015'

# Check for local files (.dat/.hea)
local_files = [record + ext for ext in ('.dat', '.hea')]
exist_mask = {f: os.path.exists(f) for f in local_files}
use_local = all(exist_mask.values())

# Determine sampling rate (fs) by reading header (local or remote)
fs = None
try:
    if use_local:
        hdr = wfdb.rdheader(record)
    else:
        hdr = wfdb.rdheader(record, pb_dir='mitdb')
    fs = getattr(hdr, 'fs', None)
    if fs is None:
        fs = hdr.__dict__.get('fs') if hasattr(hdr, '__dict__') else None
except Exception as e:
    print('Header read failed:', e)
    # fallback: try reading a tiny sample to get fs
    try:
        sig_tmp, fields_tmp = wfdb.rdsamp(record, sampfrom=0, sampto=10, pb_dir='mitdb' if not use_local else None)
        fs = fields_tmp.get('fs')
    except Exception:
        fs = None

if fs is None:
    raise RuntimeError('Could not determine sampling frequency (fs). Check record name or files.')

# samples in 5 seconds
n5 = int(5 * float(fs))

try:
    if use_local:
        print('Loading local files (first 5 s)...')
        sig, fields = wfdb.rdsamp(record, sampfrom=0, sampto=n5)
    else:
        present = [f for f, ok in exist_mask.items() if ok]
        missing = [f for f, ok in exist_mask.items() if not ok]
        if present:
            print(f"Partial local files found: {present}. Missing: {missing}.")
            print("Both .dat and .hea are required to load a local record. Attempting to download the first 5 s from PhysioNet (mitdb) instead...")
        else:
            print("Local files not found; attempting to download the first 5 s from PhysioNet (mitdb)...")
        sig, fields = wfdb.rdsamp(record, sampfrom=0, sampto=n5, pb_dir='mitdb', physical=True)

    print(f"Signal shape: {sig.shape}")
    print(f"Sampling rate: {fields.get('fs', fs)} Hz")
    print(f"Lead names: {fields.get('sig_name')}")

    # Plot the first 5 seconds (time in seconds)
    t = np.arange(sig.shape[0]) / float(fields.get('fs', fs))
    plt.figure(figsize=(12, 4))
    plt.plot(t, sig[:, 0])
    plt.title(f'ECG - Record {record} (Lead 0) — first {sig.shape[0]/float(fields.get("fs", fs)):.2f} s')
    plt.xlim(0, min(5, sig.shape[0] / float(fields.get('fs', fs))))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.show()

except Exception as e:
    print('Error loading record:', e)
    print('\nHints:')
    print('- If using local files, ensure both .dat and .hea are in this directory or provide a full path:')
    print("  e.g., wfdb.rdsamp('C:/path/to/04015')")
    print("- To download from PhysioNet, ensure the record name exists and set the correct 'pb_dir' (e.g., 'mitdb').")
    print("- If you have network issues, try using local files or check your connection.")
    print("- Install wfdb with: pip install wfdb")
    raise
