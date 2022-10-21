#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:34 2012

@author: Taimur
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import adc_lib
from time import time
from sar_ica_cl import SAR_ICA

# from adc_ideal import ADC_ideal
# from convert_cpp import convert_cpp

# Simulation parameters
FS = 1e6  # ADC sampling frequency
FIN = 0.0030059814453125e6  # input signal frequency
PTS = 65536  # points in the FFT
PLT_PTS = int(1 / (FIN / FS))
WINDOW = "no"  # WINDOW to use in the FFT
TOTAL_PTS = 3e9  # number of points to simulate

PLOT = 0  # plot at every fft
PLOT_FFT_AT_END = 1  # plot the fft result at the end of simulation
GPU = 1  # GPU=1, CPU=0
FFT_every = 10  # run FFT every X simulate/calibrate cycles

WORKERS = 256  # 256 workers, play with this to see performance differences

# ADC characteristics
RAW_BITS = 14  # number of raw bits, i.e. DAC capacitances, in the SAR ADC
VREF = 1.0  # reference voltage
C_SIGMA = 0.15  # standard deviation of DAC capacitor mismatch, e.g. 0.05 = 5%
CU = 1e-15  # unit capacitance value
COMP_OFFSET_SIGMA = 0e-3  # standard deviation of comparator offset in V
COMP_NOISE = 100e-6  # comparator noise in V
RADIX = 1.86  # radix of the SAR ADC
DELTA = 16  # pseudorandom noise amplitude in LSB
LR = 0.000005  # learning rate

adc_sar = SAR_ICA(RAW_BITS, VREF, CU, COMP_OFFSET_SIGMA, COMP_NOISE, C_SIGMA, RADIX, DELTA, LR, 0, workers=WORKERS,
                  run_on_gpu=GPU)


def main():
    out_sar = []
    # Generate input data
    x = np.linspace(0.0, (PTS - 1) * (1 / FS), num=PTS).astype(np.float32)
    y = 0.5 + 0.45 * np.sin(2 * np.pi * FIN * x)
    print("\n\nSimulating ADC...")
    sim_count = 0
    file_enob = open("enob.csv", "w")
    file_enob.write("#PTS, enob\n")
    file_error = open("error.csv", "w")
    file_error.write("#PTS, enob\n")
    run = 0
    time1_total = time()
    try:
        out_initial, elapsed = adc_sar.convert(y)
        for th in range(int(TOTAL_PTS / PTS)):
            time1_convert = time()
            out_sar, elapsed = adc_sar.convert(y)
            time2_convert = time()
            time_convert = time2_convert - time1_convert
            sim_count = sim_count + PTS
            time1_fft = time()
            if not run % FFT_every or not run:
                (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental) = adc_lib.adc_char(np.transpose([x, out_sar]), PTS,
                                                                                        0, FIN, FS, bits=14,
                                                                                        harmonics=10, lines_to_skip=0,
                                                                                        show_plot=PLOT, use_sideband=0,
                                                                                        use_window=WINDOW,
                                                                                        fit_to_full_range=0,
                                                                                        save_fft_file=False)
            run = run + 1
            time2_fft = time()
            time_fft = time2_fft - time1_fft
            print('{0:.2f} Mpts, ENOB={3:.2f}, w_delta={4:.5f} *** convert: {1:.2f}ms, fft: {2:.2f}ms'.format(
                sim_count / 1e6, time_convert * 1000, time_fft * 1000, ENOB, adc_sar.w_delta[0]))
            file_enob.write("{0}, {1}, {2}, {3}\n".format(sim_count, ENOB, SNDR, SFDR))
            file_error.write("{0}, ".format(sim_count))
            for item in range(len(adc_sar.error)):
                file_error.write("%s, " % adc_sar.error[RAW_BITS - 1 - item])
            file_error.write("\n")
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
    time2_total = time()
    time_total = time2_total - time1_total
    print('Total simulation time: {0:.2f} s'.format(time_total))

    if PLOT_FFT_AT_END:
        (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental) = adc_lib.adc_char(np.transpose([x, out_sar]), PTS, 0, FIN,
                                                                                FS, bits=14, harmonics=10,
                                                                                lines_to_skip=0, show_plot=True,
                                                                                use_sideband=0, use_window=WINDOW,
                                                                                fit_to_full_range=0,
                                                                                save_fft_file=False)

    plt.figure()
    plt.title("1 period of waveform before and after calibration")
    plt.plot(x[:PLT_PTS], out_initial[:PLT_PTS], label="Original")
    plt.plot(x[:PLT_PTS], out_sar[:PLT_PTS], label="Calibrated")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
