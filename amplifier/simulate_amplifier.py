#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:34 2012

@author: Taimur
"""

import numpy as np
import matplotlib.pyplot as plt
import adc_lib
from time import time
from amplifier_cl import amplifier

# Simulation parameters
FS = 1e6  # sampling frequency
FIN = 0.0030059814453125e6  # input signal frequency
PTS = 65536  # points in the FFT
PLT_PTS = int(1 / (FIN / FS))
WINDOW = "no"  # WINDOW to use in the FFT
TOTAL_PTS = 1e9  # number of points to simulate

PLOT = 0  # plot at every fft
PLOT_FFT_AT_END = 1  # plot the fft result at the end of simulation
GPU = 1  # GPU=1, CPU=0
TOTAL_PTS = 0.3e9
FFT_every = 10  # run FFT every X simulate/calibrate cycles

WORKERS = 256  # 256 workers, play with this to see performance differences

# Amplifier characteristics
ORDER = 5  # nonlinearity model order
VREF = 1.0  # amplitude of the input waveform
DELTA = 0.1 * VREF  # injected pseudorandom noise amplitude
LR = 0.0000001  # learning rate

# Declare amplifier
amp = amplifier(delta=DELTA, order=ORDER + 1, lr=LR, run_on_gpu=GPU, workers=WORKERS)
# amplifier nonlinearity coefficients
amp.amp_coeff = np.array([0.0, 1.0, 0.01, -0.15, 0.0, 0.0])


def main():
    out = []
    # Generate input data
    x = np.linspace(0.0, (PTS - 1) * (1 / FS), num=PTS).astype(np.float32)
    y = VREF * np.sin(2 * np.pi * FIN * x).astype(np.float32)
    print("\n\nSimulating ADC...")
    sim_count = 0
    file_thd = open("thd.csv", "w")
    file_thd.write("#PTS, thd\n")
    file_coefficients = open("coefficients.csv", "w")
    file_coefficients.write("#PTS, enob\n")
    initial = 1
    run = 0
    time1_total = time()
    try:
        for th in range(int(TOTAL_PTS / PTS)):
            time1_convert = time()
            out, elapsed = amp.run(y)
            time2_convert = time()
            time_convert = time2_convert - time1_convert
            sim_count = sim_count + PTS
            time1_fft = time()
            if not run % FFT_every or not run:
                (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental) = adc_lib.adc_char(np.transpose([x, out]), PTS, 0,
                                                                                        FIN, FS, bits=14,
                                                                                        harmonics=10, lines_to_skip=0,
                                                                                        show_plot=PLOT, use_sideband=0,
                                                                                        use_window=WINDOW,
                                                                                        fit_to_full_range=0,
                                                                                        save_fft_file=False)
            run = run + 1
            time2_fft = time()
            time_fft = time2_fft - time1_fft
            print('{0:.2f} Mpts, THD={3:.2f} *** convert: {1:.2f}ms, fft: {2:.2f}ms'.format(sim_count / 1e6,
                                                                                            time_convert * 1000,
                                                                                            time_fft * 1000, THD))
            file_thd.write("{0}, {1}\n".format(sim_count, THD))
            file_coefficients.write("{0}, ".format(sim_count))
            for item in range(len(amp.b)):
                file_coefficients.write("%s, " % amp.b[item])
            file_coefficients.write("\n")
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")

    time2_total = time()
    time_total = time2_total - time1_total
    print('Total simulation time: {0:.2f} s'.format(time_total))

    if PLOT_FFT_AT_END:
        (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental) = adc_lib.adc_char(np.transpose([x, out]), PTS, 0, FIN,
                                                                                FS, bits=14, harmonics=10,
                                                                                lines_to_skip=0, show_plot=True,
                                                                                use_sideband=0, use_window=WINDOW,
                                                                                fit_to_full_range=0,
                                                                                save_fft_file=False)
    plt.figure()
    plt.title("1 period of waveform before and after calibration")
    plt.plot(x[:PLT_PTS], y[:PLT_PTS], label="Original")
    plt.plot(x[:PLT_PTS], out[:PLT_PTS], label="Calibrated")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
