"""Export the zea/Open-H-RF dataset to the DASREF02 binary format."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the Blosc filter used by these files
import numpy as np


MAGIC = b"DASREF02"
HEADER = struct.Struct("<8sIIIddddd")  # magic, n_tx, n_samples, n_rx, fs, sound_speed, fc, t_peak, f_num


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Input zea HDF5 dataset")
    parser.add_argument("output", type=Path, help="Output DASREF02 bundle")
    parser.add_argument("--fnum", type=float, default=1.75, help="Receive f-number for dynamic aperture (default: 1.75)")
    parser.add_argument("--tpeak", type=float, default=None, help="Pulse peak time offset in seconds (default: 1/fc)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as h5:
        raw_ds = h5["tracks/track_0/data/raw_data"]
        if len(raw_ds.shape) != 5 or raw_ds.shape[0] != 1 or raw_ds.shape[-1] != 1:
            raise ValueError(f"Expected one frame and one real RF channel, got {raw_ds.shape}")
        _, n_tx, n_samples, n_rx, _ = raw_ds.shape
        scan = h5["tracks/track_0/scan"]
        probe = h5["probe"]

        geometry = np.asarray(probe["probe_geometry"][()], dtype="<f8")
        initial_times = np.asarray(scan["initial_times"][()], dtype="<f8")
        t0_delays = np.asarray(scan["t0_delays"][()], dtype="<f8")
        tx_apodizations = np.asarray(scan["tx_apodizations"][()], dtype="<f8")
        fs = float(scan["sampling_frequency"][()])
        sound_speed = float(scan["sound_speed"][()])

        if "demodulation_frequency" in scan:
            fc = float(scan["demodulation_frequency"][()])
        elif "center_frequency" in scan:
            fc = float(scan["center_frequency"][()])
        else:
            fc = 6.0e6

        t_peak = float(1.0 / fc) if args.tpeak is None else float(args.tpeak)
        f_num = float(args.fnum)

        if geometry.shape != (n_rx, 3):
            raise ValueError(f"Probe geometry {geometry.shape} does not match {n_rx} receive channels")
        if initial_times.shape != (n_tx,) or t0_delays.shape != (n_tx, n_rx):
            raise ValueError("Transmit timing arrays do not match raw data dimensions")
        if tx_apodizations.shape != (n_tx, n_rx):
            raise ValueError("Transmit apodizations do not match raw data dimensions")
        if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(t0_delays)):
            raise ValueError("Geometry and transmit delays must be finite")

        with args.output.open("wb") as out:
            out.write(HEADER.pack(MAGIC, n_tx, n_samples, n_rx, fs, sound_speed, fc, t_peak, f_num))
            for arr in (geometry, initial_times, t0_delays, tx_apodizations):
                np.ascontiguousarray(arr).tofile(out)
            # Keep the HDF5 ordering: transmit, axial sample, receive element.
            np.asarray(raw_ds[0, :, :, :, 0], dtype="<f4").tofile(out)

    print(f"Wrote {args.output} ({n_tx} tx x {n_samples} samples x {n_rx} rx, fc={fc/1e6:.1f}MHz, t_peak={t_peak:.4e}s, f_num={f_num})")


if __name__ == "__main__":
    main()
