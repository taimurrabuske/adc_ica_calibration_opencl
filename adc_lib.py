import matplotlib.pyplot as plt
from numpy import loadtxt,savetxt,log10,array,arange,log2
from numpy.fft import fft, fftshift
from scipy.signal import get_window
from scipy.optimize import curve_fit
import numpy as np


from numpy.fft import rfft
from numpy import asarray, argmax, mean, diff, log, copy
from scipy.signal import correlate, kaiser, decimate





def adc_char(data, points=1024, start=0, inputfreq=1, samplingrate=0.1, bits=12, harmonics=10, lines_to_skip=0, csv=1,
             find_fundamental=1, use_sideband=0, use_window='no', show_plot=0, fit_to_full_range=0, osr=1,
             one_column_file=0, save_fft_file=1):

    if type(data) == str:
        filename=data
        if csv:
            data = loadtxt(filename, delimiter=',', skiprows=lines_to_skip)
        else:
            data = loadtxt(filename, skiprows=lines_to_skip)

    if one_column_file:
        d = data
    else:
        t = data[:, 0]
        d = data[:, 1]

    # Eliminate samples before analysis starting time
    d = d[start:start + points]

    d_unwindowed = d

    # window
    if use_window != "no":
        d = d * get_window(use_window, len(d))

    out = d

    # Input signal FFT vector index
    fundamental_index = int(inputfreq * points / samplingrate)

    top = max(out)
    bottom = min(out)

    # FFT with negative frequencies. fftshift centers DC to 0 on the plot
    outfreq_doublesided = fftshift(abs(fft(out)))

    # We're just interested in positive frequencies.
    # outfreq is the frequency domain signal
    outfreq = outfreq_doublesided[int(points / 2):points]

    # If the ADC is a SD, take only the important part of the spectrum
    if osr != 1:
        pts = points / osr
        outfreq = outfreq[0:int(len(outfreq) / osr)]
    else:
        pts = points

    # outfreq^2 is the output power.
    outfreq = outfreq * outfreq

    fundamental = "Not found"
    if find_fundamental:
        # sort indexes backwards
        order = outfreq.argsort()[::-1]
        k = 0
        if use_sideband == 0:
            sideband_dc = 1
        else:
            sideband_dc = use_sideband
        while order[k] < sideband_dc:
            k = k + 1
        fundamental_index = int(order[k])
        fundamental = fundamental_index * samplingrate / points

    # fundamental power
    if use_sideband:
        Ps = outfreq[fundamental_index - use_sideband:fundamental_index + use_sideband].sum(axis=0)
    else:
        Ps = outfreq[fundamental_index]

    # DC signal power
    if use_sideband:
        Pdc = outfreq[0:use_sideband].sum()
    else:
        Pdc = outfreq[0]

    # index for the harmonics
    harm_index = []

    # Bring back harmonics indexes to the FFT freq. range.
    for i in range(2, harmonics + 2):
        if 1.0 * i * fundamental_index / pts % 1 > 0.5:
            j = i * fundamental_index
            while (j >= pts):
                j = j - pts
            harm_index.append(pts - j)
        else:
            j = i * fundamental_index
            while (j >= pts):
                j = j - pts
            harm_index.append(j
        # numpt=data_size-code_count[0]-code_count[-1]
        # # code_count=code_count[int(forget):-int(forget)]
        #
        # vin=range(len(code_count))-np.ones(len(code_count))*(len(code_count)/2)+0.5
        # sin2ramp=1./(3.141592*np.sqrt(A**2*np.ones(len(code_count))-vin**2));

)

    # sum harmonics power.
    D = []
    for i in range(0, harmonics):
        if use_sideband:
            D.append(outfreq[int(harm_index[i] - use_sideband):int(harm_index[i] + use_sideband)].sum())
        else:
            D.append(outfreq[int(harm_index[i])].sum())

    Pd = array(D).sum(axis=0)
    # Noise power is equal to everything but dc, noise, harmonics and fund.
    Pn = outfreq.sum(axis=0) - Ps - Pd - Pdc
    # Calculate performance metrics
    SNR = 10 * log10(Ps / Pn)
    SNDR = 10 * log10(Ps / (Pn + Pd))
    ENOB = (SNDR - 1.76) / 6.02
    if fit_to_full_range:
        ENOB = ENOB * (bits / log2(top - bottom))
    SFDR = 10 * log10(Ps / max(D))
    THD = 10 * log10(Pd / Ps)
    Harmonics = array(D) / Ps
    # If harmonics are zero, assign -200dB, to avoid division by zero in log10
    for harm in range(len(Harmonics)):
        if Harmonics[harm] == 0:
            Harmonics[harm] = 1e-20
    Harmonics = 10 * log10(Harmonics)

    if show_plot:
        plt.subplot(2, 1, 1)
        plt.plot(d_unwindowed)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        n = pts  # length of the signal
        k = arange(n)
        T = n / samplingrate * osr
        frq = k / T  # two sides frequency range
        frq = frq[list(range(int(n / 2)))] / 1000  # one side frequency range
        y = abs(outfreq ** 0.5 / outfreq[fundamental_index] ** 0.5)
        # if y=0, assign -200 to avoid division by zero in plot
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = 1e-10
        y_db = 20 * log10(y)
        if save_fft_file:
            savetxt("fft.csv", array([frq, y_db]).transpose(), delimiter=",",
                    header="#format table ## [WaveView Analyzer] saved\nXVAL,Y_DB")
        plt.plot(frq, y_db)
        plt.xlabel('Freq (kHz)')
        plt.ylabel('|Y(freq)|')
        plt.show()

    return (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental)


def adc_char1(data, points=1024, start=0, inputfreq=1, samplingrate=0.1, bits=12, harmonics=10, lines_to_skip=0, csv=1,
         find_fundamental=1, use_sideband=0, use_window=None, show_plot=0, fit_to_full_range=0, osr=1,
         one_column_file=0, save_fft_file=1):
    return

class ADC_char(object):
    def __init__(self,data=None, points=1024, start=0, inputfreq=0.1, samplingrate=1, bits=12, harmonics=10, lines_to_skip=0, csv=1,
             find_fundamental=1, use_sideband=0, use_window=None, show_plot=0, fit_to_full_range=0, osr=1,
             one_column_file=0, save_fft_file=1):
        self.t = np.array([])
        self.d = np.array([])
        if data:
            self.load_data(data, one_column_file, csv, lines_to_skip)
        self.points = points
        self.start = start
        self.FIN = inputfreq
        self.samplingrate = samplingrate
        self.bits = bits
        self.harmonics = harmonics
        self.window = use_window
        self.sideband = use_sideband
        # self.fit_to_full_range = fit_to_full_range
        self.osr = osr
        self.find_fundamental = find_fundamental
        self.start = start

        self.SNR = None
        self.SNDR = None
        self.ENOB = None
        self.ENOB_fit = None
        self.SFDR = None
        self.THD = None
        #    Harmonics=10*log10(array(D)/Ps)

        self.harm_index = []
        self.harm_power = []
        self.Harmonics = np.array([])



        self.t = np.array([0])
        self.x = np.array([0])
        self.d = np.array([0])

        self.f = np.array([0,])
        self.d_f = np.array([0,])
        self.p_f = np.array([])
        self.d_db_f = np.array([])
        self.d_windowed = np.array([])

        self.static_skip = 1
        self.static_x = np.array([])
        self.static_count = np.array([])
        self.static_fit = np.array([])
        self.DNL = np.array([])
        self.INL = np.array([])


        self.x_wrap = np.array([])
        self.x_order = np.array([])
        self.d_wrap = np.array([])

    def do(self):
        self.get_window()
        self.fft()
        if self.find_fundamental:
            self.compute_fundamental()
        self.powers()

    def load_data(self, data, one_column_file=False, csv=True, lines_to_skip=0):
        if type(data) == str:
            filename=data
            if csv:
                data = loadtxt(filename, delimiter=',', skiprows=lines_to_skip)
            else:
                data = loadtxt(filename, skiprows=lines_to_skip)

        if one_column_file:
            self.d = data
        else:
            self.t = data[:, 0]
            self.d = data[:, 1]
        self.x = np.array(range(len(self.d)))

    def get_window(self):
        # Eliminate samples before analysis starting time
        d = self.d[self.start:self.start + self.points]
        # window
        if self.window:
            self.d_windowed = d * get_window(self.window, len(d))
        else:
            self.d_windowed = d


    def fft(self):
        out = self.d_windowed

        # FFT with negative frequencies. fftshift centers DC to 0 on the plot
        # outfreq_doublesided = fftshift(abs(fft(out)))
        outfreq_doublesided = abs(fft(out))

        # We're just interested in positive frequencies.
        # outfreq is the frequency domain signal
        # outfreq = outfreq_doublesided[int(self.points / 2):self.points]
        outfreq = outfreq_doublesided[0:int(self.points / 2)]
        # outfreq^2 is the output power.


        # If the ADC is a SD, take only the important part of the spectrum
        if self.osr != 1:
            self.pts = self.points / self.osr
            outfreq = outfreq[0:int(len(outfreq) / self.osr)]
        else:
            self.pts = self.points
        self.d_f = outfreq
        self.p_f = outfreq * outfreq
        # Input signal FFT vector index
        self.fin_index = int(self.FIN * self.points / self.samplingrate)

    def compute_fundamental(self):
        # sort indexes backwards
        order = self.d_f.argsort()[::-1]
        k = 0
        if self.sideband == 0:
            sideband_dc = 1
        else:
            sideband_dc = self.sideband
        while order[k] < sideband_dc:
            k = k + 1
        self.fin_index = int(order[k])
        self.FIN = self.fin_index * self.samplingrate / self.points

    def powers(self):

        self.d_f = abs(self.d_f/self.d_f[self.fin_index])
        self.p_f = abs(self.p_f/self.p_f[self.fin_index])

        top = max(self.d)
        bottom = min(self.d)
        # fundamental power


        D = []
        # index for the harmonics
        self.harm_index = []
        self.harm_power = []

        # Bring back harmonics indexes to the FFT freq. range.
        for i in range(0, self.harmonics+1):
            if 1.0 * i * self.fin_index / self.pts % 1 > 0.5:
                j = i * self.fin_index
                while (j >= self.pts):
                    j = j - self.pts
                self.harm_index.append(self.pts - j)
            else:
                j = i * self.fin_index
                while (j >= self.pts):
                    j = j - self.pts
                self.harm_index.append(j)

            if self.sideband:
                # sum harmonics power.
                i_min = max(int(self.harm_index[i])-self.sideband, 0)
                i_max = min(int(self.harm_index[i])+self.sideband, int(self.pts/2))
                self.harm_power.append(self.p_f[i_min:i_max].sum())
            else:
                self.harm_power.append(self.p_f[int(self.harm_index[i])])

        Pdc = self.harm_power[0]
        Ps = self.harm_power[1]
        Pd = array(self.harm_power[2:]).sum()
        # Noise power is equal to everything but dc, noise, harmonics and fund.
        Pn = self.p_f.sum(axis=0) - Ps - Pd - Pdc
        # Calculate performance metrics
        self.SNR = 10 * log10(Ps / Pn)
        self.SNDR = 10 * log10(Ps / (Pn + Pd))
        self.ENOB = (self.SNDR - 1.76) / 6.02
        self.ENOB_fit = self.ENOB * (self.bits / log2(top - bottom))
        self.SFDR = 10 * log10(Ps / max(self.harm_power[2:]))
        self.THD = 10 * log10(Pd / Ps)
        #    Harmonics=10*log10(array(D)/Ps)
        Harmonics = array(self.harm_power) / Ps
        # If harmonics are zero, assign -200dB, to avoid division by zero in log10
        for harm in range(len(Harmonics)):
            if Harmonics[harm] == 0:
                Harmonics[harm] = 1e-20
        self.Harmonics = 10 * log10(Harmonics)

        k = arange(int(self.pts/2))
        T = self.pts / self.samplingrate * self.osr
        self.f = k / T  # two sides frequency range
        # y = abs(self.d_f/self.d_f[self.fin_index])
        # if y=0, assign -200 to avoid division by zero in plot
        y = self.d_f

        for i in range(len(y)):
            if y[i] == 0:
                y[i] = 1e-10
        y_db = 20 * log10(y)
        self.d_db_f = y_db

    #return (SNR, SNDR, ENOB, SFDR, THD, Harmonics, fundamental)

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.d)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(self.f, self.d_db_f)
        plt.xlabel('Freq (kHz)')
        plt.ylabel('|Y(freq)|')
        plt.show()

    def save_fft_file(self):
        savetxt("fft.csv", array([self.f, self.d_db_f]).transpose(), delimiter=",",
                header="#format table ## [WaveView Analyzer] saved\nXVAL,Y_DB")


    def dnl_inl(self):
        data = self.d
        x_start = int(data.min()) + self.static_skip
        x_stop = int(data.max()) - self.static_skip


        x = np.linspace(0,2**self.bits-1, 2**self.bits)
        code_count=np.zeros(2**self.bits)
        for i in data:
            code_count[int(i)]=code_count[int(i)]+1

        code_count = code_count[x_start:x_stop]
        x = x[x_start:x_stop]
        num_pts = code_count.sum()

        def p_sine(x, a, b, num_pts):
            return num_pts*1/(np.pi*np.sqrt((x-a)*(b-x)))
        popt, pcov = curve_fit(p_sine, x, code_count, p0=(0,2**self.bits, num_pts))
        a, b, num_pts = popt

        dnl_1 = (code_count/p_sine(x,a,b,num_pts))
        dnl = dnl_1-dnl_1.mean()
        dnl=dnl[np.isfinite(dnl)]

        self.static_x = x
        self.static_count = code_count
        self.static_fit = p_sine(self.static_x, a,b, num_pts)
        self.DNL = dnl
        self.INL=np.cumsum(dnl)


    def wrap(self):
        from coherent_sampling import is_coherent
        if not is_coherent(self.FIN, self.samplingrate, self.points):
            print("Not coherent")
        self.x_wrap = self.x % (self.samplingrate/self.FIN)
        self.x_order = np.argsort(self.x_wrap)
        self.d_wrap = self.d[self.x_order]

