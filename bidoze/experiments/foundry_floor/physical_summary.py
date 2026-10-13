"""Summarize verified floor windows with explicit PSD conventions and physical RMS."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import senpy
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    done=json.loads((a.out/"complete.json").read_text());assert done["windows"]==43
    data=pd.read_csv(a.out/"summary.csv");assert len(data)==43 and data.inputs_unchanged.all()
    assert data.groupby("device").size().to_dict()=={"Eric-Galaxy":21,"Eric-iPhone":22}
    data["nufft_trace_one_sided_db_q10"]=data.nufft_sum_median_bin_db_q10+10*np.log10(2)
    # Band 2-6 Hz excludes DC and actual Nyquist (25 Hz), so folding doubles all bins.
    data["nufft_physical_vector_band_rms_mg_q10"]=data.nufft_sum_rms_at_quietest_mg_q10*np.sqrt(2)
    data.to_csv(a.out/"summary_physical.csv",index=False)
    metrics=["legacy_mean_axis_db_q10","legacy_mean_axis_rms_mg_q10",
        "corrected_sum_mean_bin_db_q10","corrected_sum_rms_mg_q10",
        "nufft_sum_median_bin_db_q10","nufft_trace_one_sided_db_q10",
        "nufft_physical_vector_band_rms_mg_q10","selected_frames_below_3mg_fraction"]
    data.groupby("device")[metrics].agg(["median","min","max"]).to_csv(a.out/"device_summary_physical.csv")
    rng=np.random.default_rng(713);t=np.arange(0.,120.,.02);x=rng.normal(0,.001,len(t))
    got=senpy.compute_nufft_spectrogram(t,x,10,8,kind="psd",origin_s=0,empty_windows="keep")
    f,expected=welch(x,fs=50,nperseg=500,noverlap=400,detrend="constant")
    band=(f>=2)&(f<=6);other=(got.frequencies>=2)&(got.frequencies<=6)
    ratio=np.mean(got.Sxx[:,other])/np.mean(expected[band]);assert .48<ratio<.52
    check=dict(seed=713,duration_seconds=120,fs=50,axis_sigma_mg=1,variance=np.var(x),
        welch_one_sided_integral=np.trapezoid(expected,f),
        senpy_positive_half_integral=np.mean(np.trapezoid(got.Sxx,got.frequencies,axis=1)),
        band_density_ratio=float(ratio),senpy_version=getattr(senpy,"__version__","unpackaged"),senpy_api_sha256=hashlib.sha256(Path(senpy.__file__).with_name("api.py").read_bytes()).hexdigest(),
        meaning="Senpy intentionally omits scipy's one-sided x2; model inputs remain unchanged.")
    (a.out/"density_convention_check.json").write_text(json.dumps(check,indent=2)+"\n")
    protocol=json.loads((a.out/"protocol.json").read_text())
    protocol.update(senpy_density="positive-frequency half of two-sided density; source contract intentional",
        rms="physical 2-6 Hz vector RMS folds both frequency halves: sqrt(2 * integral_positive_band)",
        legacy_rms="root mean-axis variance; multiply by sqrt(3) for the vector RMS",
        physical_rms_is_noise_only=False,model_inputs_rescaled=False,
        additive_signal="Independent additive component / floor RMS = sqrt(10**(dB/10)-1); total RMS rise differs.",
        note="Do not convert the separately measured magnitude-channel wake/sleep contrast using a trace floor.")
    (a.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (a.out/"physical_summary_complete.json").write_text(json.dumps(dict(complete=True,windows=43))+"\n")
    print(data.groupby("device")[metrics].median().to_string())
if __name__=="__main__":main()
