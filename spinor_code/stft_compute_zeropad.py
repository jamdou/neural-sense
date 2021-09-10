from __future__ import division
from faraday_aux import *   # the usual suspects
from numpy.fft import fft   # NB: scipy fftpack hemorages memory
from pylab import *
from lyse import Run, path, spinning_top
#import peakutils
import lmfit
import os 
import sys
import scipy

from scipy.signal import hilbert, filtfilt, lfilter
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.optimize import curve_fit
from scipy.fftpack import fft, fftfreq

# Hamish's anti noise calibration 
noise_calibration = False

# stft arguments
tstart = None
tmax = 20
# tmax = 0.2
resample = 2e-4 #0.2e-3 # time-domain spacing of STFT slices
window = 3e-3 #3e-3 #18e-3      # STFT window size
freq_resample = 6
estimate_snr = False
import_ref = False   # Import reference trace from Faraday channel B and spectrogram it;
# stft_range = [0.5, 3]
# stft_range = [3, 6]
stft_range = [2, 4] #[6, 9] #[2, 4]
ref_stft_range = [4, 7]
# stft_range = [1.5, 3.5]

# Larmor estimation
estimate_larmor = False
larmor_guess_start = 15e-3
# larmor_guess_end   = 35e-3
larmor_guess_end   = 75e-3
# larmor_guess_end   = 165e-3
# larmor_guess_end   = 420e-3



# Plot parameters
# frange = 100e3       # frequency around around sp_center_freq to plot
frange = 20e3 #5e-3       # frequency around around sp_center_freq to plot
# frange = 30e3       # frequency around around sp_center_freq to plot
#frange = 20e3       # frequency around around sp_center_freq to plot#
# frange = 200e3       # frequency around around sp_center_freq to plot
save_png = False     # save a PNG of the STFT
# center_freq_global = 'sp_center_freq'
field_calibration = False
save_results = False
# Single shot Rabi spectroscopy params
single_shot_rabi_spec = False

if not spinning_top:
    df = lyse.data()
    path = df['filepath'][0]
dataset = lyse.data(path)
run = Run(path)

try:
    if import_ref:
        t_ref, V_ref, globs_ref = get_alazar_trace(path, tmax=tmax, trace_location='/data/traces/alazar_faraday/channelB')
    t, V, globs = get_alazar_trace(path, tmax=tmax)
    trace_exists = True
except KeyError:
    print('No alazar_trace in ' + os.path.split(path)[-1])
    trace_exists = False

if not trace_exists:
    try:
        t, V, globs = get_raw_trace(path, tmax=tmax)
        trace_exists = True
    except KeyError:
        print('No raw_trace in ' + os.path.split(path)[-1])
        trace_exists = False
else:
    pass

if trace_exists:
    if 'sp_calib_sweep' in globs and globs['sp_calib_sweep']:
        center_freq_global = 'sp_calib_center_freq'
    else:
        center_freq_global = 'sp_center_freq'
    R = int(globs['acquisition_rate'])
    fsp = globs[center_freq_global]*1e6
    # fsp = 5.25*1e6
    # fsp = 725e3
    print('rf frequency = {:.3f} kHz'.format(fsp/1e3))

    fftsize = int(window * R * freq_resample)
    # fftsize = 3*102400    # size of the padded FFT
    # fftsize = R//10000    # 100 Hz bins in freq space
    print('Oversampling frequencies by {:d}. FFT size is {:d}'.format(freq_resample, fftsize))
    assert fftsize >= window * R

    #################### COMPUTE STFT ####################      
    def stft(signal, dt, slices, size=None, subset=None, window='hamming', t0=0):
        # window is either a function or string (name of one of scipy's windows)
        if not callable(window):
            import scipy.signal.windows
            window = getattr(scipy.signal.windows, window)
            print(window)
        # estimate size of individual FFTs
        if size is None:
            size = len(signal)//slices
        # separation in frequency space
        df = 1.0/dt/fftsize
        print('df = {:.1f} Hz; size = {:d}'.format(df, size))
        if subset is None:
            # return entire Nyquist range (memory intensive!)
            f1, f2 = 0, size//2
        else:
            # return only subset of frequency space
            f1, f2 = int(subset[0]/df+0.5), int(subset[1]/df+0.5)
        # create the window
        W = window(size)
        print(W)
        print(len(W))
        FW = fft(W)
        # allocate empty space for the result
        res = np.empty((slices, f2-f1+1))
        t = np.empty(slices)
        for i, j in enumerate(np.linspace(-size/2,len(signal),slices)):
            # compute FFT from within sample space, keeping only the region of interest
            j = int(j+0.5)
            t[i] = (j + size/2.)*dt + t0
            V = signal[j:j+size]
            if j < 0:
                V = np.hstack((np.zeros((-j,)),signal[:size+j]))
            elif j > len(signal)-size:
                V = np.hstack((V,np.zeros((j+size-len(signal),),)))
            V = V * W
            #print(len(V))
            V = np.hstack((V,np.zeros((fftsize-size,))))
            FV = fft(V)
            FV = FV[f1:f2+1]
            res[i,:] = abs(FV)**2
        return t, res

    slices = int(len(V)/R/resample)
    print('Interval = {:.1f} ms; Window = {:.1f} ms; Slices = {:d}'.format(1e3*resample, 1e3*window, slices))
    assert slices > 10

    subset = (fsp-frange/2, fsp+frange/2)
    # subset=(fsp, fsp+1.5*frange)
    # subset = (0, fsp+frange/2)
    # subset = (fsp-frange/2, 4.5e6)
    #subset = (800e3, 1200e3)
    print('Computing STFT over frequency range ({:.1f}, {:.1f}) kHz'.format(subset[0]/1e3, subset[1]/1e3))
    t1, res = stft(signal=V, dt=1/R, slices=slices, size=int(window*R), subset=subset, t0=t[0])
    f1 = np.linspace(subset[0], subset[1], res.shape[1])
    df = f1[1]-f1[0]
    print(t1)
    print(res)
    ############## CREATE REF TRACE STFT ##############
    # Only do this if we have the reference being imported
    if import_ref:
        t1_ref, res_ref = stft(signal=V_ref, dt=1/R, slices=slices, size=int(window*R), subset=subset, t0=t[0])
        f1_ref = np.linspace(subset[0], subset[1], res_ref.shape[1])
        df_ref = f1_ref[1]-f1_ref[0]
        res_ref /= window
        res_ref /= 80

    ############## ESTIMATE LARMOR FREQUENCY ##############
    # Time-averaged spectrogram power
    if estimate_larmor:
        t_filter = (t1 > larmor_guess_start) * (t1 < larmor_guess_end)
        res_f = np.sum(res[t_filter]**2, 0)
        res_f = (res_f-res_f.min())**2
        fL_guess = np.dot(res_f, f1)/res_f.sum()   # i am a biased estimation (heavily weighted towards start of signal)
        print('Larmor estimate: {:.3f} kHz'.format(fL_guess/1e3))

    #################### ESTIMATE SNR #####################
    # Frequency-averaged spectrogram amplitude
    res_t = np.sum(res, 1)
    res_t -= res_t.min()

    if estimate_snr:    
        t_hobbs_settle = globs['hobbs_settle']
        t_pre_tip_wait = t_hobbs_settle + globs['faraday_pre_tip_wait']
        print(t_hobbs_settle)
        print(t_pre_tip_wait)
        t_faraday_hold = t_pre_tip_wait + globs['faraday_hold_time']
        A_electronic = res_t[t1 < t_hobbs_settle].mean()
        
        A_photon = res_t[(t1 > t_hobbs_settle) * (t1 < (t_pre_tip_wait-window/2))].mean()
        A_atoms = res_t[(t1 > t_pre_tip_wait) * (t1 < t_faraday_hold)].max()
        SNR_photon = A_photon / A_electronic
        SNR_atoms = A_atoms / A_photon

    #################### SINGLE SHOT RABI SPECTROSCOPY #####################
    if single_shot_rabi_spec and globs['magnetom_ac_sweep']:
        # Import relevant globals
        f_center = globs['sp_center_freq']*1e6
        f_mag_center = globs['magnetom_ac_frequency']
        f_mag_bandwidth = globs['magnetom_ac_sweep_bandwidth']
        mag_t_start = globs['hobbs_settle'] + globs['magnetom_ac_wait']
        mag_t_finish = mag_t_start + globs['magnetom_ac_duration']

        def find_peaks(f, y, absthreshold=40000, min_dist=10):
            indexes = peakutils.indexes(y, thres=absthreshold/max(y), min_dist=min_dist)
            peaks = peakutils.interpolate(f, y, ind=indexes, width=min_dist)
            return [p for p in peaks if p >= f.min() and p <= f.max()]

        # Isolate desired sideband data
        # res_smaller = res[t1 > mag_t_start and t1 < mag_t_finish, f1 > 483e3 and f1 < 485e3]
        res_smaller = res[np.logical_and(t1 > mag_t_start, t1 < mag_t_finish), :]
        res_smaller = res_smaller[:, np.logical_and(f1 > 483e3, f1 < 485e3)]
        # t_upper = t1[t1 > mag_t_start and t1 < mag_t_finish]
        t_upper = t1[np.logical_and(t1 > mag_t_start, t1 < mag_t_finish)]
        # t_upper = t1[t1 > mag_t_start]
        # t_upper = t_upper[t_upper < mag_t_finish]
        # f_upper = f1[f1 > 483e3 and f1 < 485e3]
        f_upper = f1[np.logical_and(f1 > 483e3, f1 < 485e3)]
        # f_upper = f1[f1 > 483e3]
        # f_upper = f_upper[f_upper < 485e3]
        f_peaks = f_upper[np.argmax(res_smaller, axis = 1)]

        # for i in range(len(f1)):
        #     if f1[i] > 483e3 and f1[i] < 485e3:
        #         for j in range(len(t1)):
        #             if t1[j] > mag_t_start and t1[j] < mag_t_finish:
        #                 if np.log(res[j,f1 > 483e3:f1 < 485e3]) > 2:
        #                     f_upper.append(f1[i])
        #                     t_upper.append(t1[j])

        # # Convert resulting arrays to numpy
        # f_upper = np.array(f_upper)
        # t_upper = np.array(t_upper)

        # Convert times on x axis into detuned frequencies, and subtract offset from FFT freq
        f_sweep_init = f_mag_center - f_mag_bandwidth/2
        fi = (t_upper - mag_t_start)/globs['magnetom_ac_duration'] * f_mag_bandwidth + f_sweep_init
        f_peaks -= f_center

        # Plot calibration figure of time to freqency for comparison
        plt.figure()
        plt.plot(t_upper, fi)
        plt.xlabel('faraday signal time (s)')
        plt.ylabel('Signal frequency (Hz)')
        plt.title('Time to Frequency X axis conversions')

        # Perform averages in intervals of 5 to reduce numbers of points
        # f_peaks_red = np.mean(f_peaks[:(len(f_peaks)//11)*11].reshape(-1,11), axis=1)
        # fi_red = fi[5::11]

        # Define fit function
        def rabiSpec(f_i, rabi, f_0):
            return np.sqrt(rabi**2 + (f_0 - f_i)**2)

        # Define fit function
        def rabiSpecLin(f_i, rabi, f_0, grad, f_c):
            return np.sqrt(rabi**2 + ((f_0 - f_i)*(grad + 1))**2)
        # # Define fit function v2
        # def rabiSpecLin(f_i, rabi, f_0, grad, f_c):
        #     return np.sqrt(rabi**2 + (f_0*(1 + grad) - f_i * (1 + grad))**2)

        # Set up fit with guess params and perform fit
        model = lmfit.Model(rabiSpec)
        model_params = model.make_params(rabi = 1500, f_0 = f_mag_center)
        result = model.fit(f_peaks, model_params, f_i = fi)

        # # Set up fit with extra linear grad underlying
        # model2 = lmfit.Model(rabiSpecLin)
        # model2_params = model2.make_params(rabi = 1500, f_0 = f_mag_center, grad = 1, f_c = f_mag_center)
        # result2 = model2.fit(f_peaks, model2_params, f_i = fi)

        # Output fit report
        print(result.fit_report())
        # print(result2.fit_report())

        # Plot fit result
        plt.figure('rabiSpectroscopy', figsize=(10,8))
        plt.plot(fi, f_peaks, linestyle = 'none', marker = 'o', label = 'data')
        # plt.plot(fi_red, f_peaks_red, 'bo', label = 'data')
        plt.plot(fi, result.best_fit, linestyle = 'solid', linewidth = 4.0, label = 'fit')
        # plt.plot(fi, result2.best_fit, 'r-', linewidth = 4.0, label = 'fit w/ offset')
        plt.grid()
        # plt.title(seq_str + '\n Rabi spectroscopy: upper sideband fit')
        plt.title('Rabi spectroscopy: upper sideband fit')
        plt.ylabel('Sideband frequency (Hz)')
        plt.xlabel('AC Signal frequency (Hz)')
        plt.legend()
        plt.show()

    #################### SAVE RESULTS #####################
    if save_results:
        res /= window
        res /= 80
        try:
            run = Run(path)
            run.set_group('faraday')
            run.save_result_array('stft', res)
            oversample = window/resample # This is the time-oversampling factor
            run.save_results_dict({
                    't0': t1[0], 'dt': t1[1]-t1[0],
                    'f0': f1[0], 'df': f1[1]-f1[0],
                    'tmax': t1[-1] if tmax is None else tmax, 
                    'oversample': oversample,
                    'resample': resample, 'window': window,
                    'Vrange': max(V)-min(V),
                    'freq_resample': freq_resample
                })
            run.save_result('larmor_guess', fL_guess)
            if estimate_snr:
                run.save_results_dict({'SNR_atom': SNR_atoms, 'SNR_photon': SNR_photon})
        except Exception as e:
            print('>> Failed to save STFT')
            print(e)

    print("calc_done")
    ###################### PLOTTING #######################
    scale = 10e2
    shot_id = os.path.split(path)[-1].replace('.h5', '')
    folder = os.path.dirname(path)
    plt.figure('Faraday spectrogram',figsize=(9,4.787))
    # 2nd Half of 600ms spectrogram settings res[1200:-1], t1[1450:-1] 
    plot_stft(res[:-1,], t1[:-1], f1, range=stft_range, png=None)#, cmap='gray_r')
    # axhline(fL_guess/1e3, c='w', ls='--', lw=1)
    # axhline(fsp/1e3, c='m', ls='--', lw=1)
    # if not (field_calibration and estimate_snr and globs['sp_B_sweep']):
    #     axhline(fsp/1e3-2, c='lime', ls='--', lw=1)
    #     axhline(fsp/1e3+2, c='lime', ls='--', lw=1)
    # else:
    #     tr = linspace(t_pre_tip_wait, t_faraday_hold, 100)
    #     By0r = linspace(globs['sp_B_sweep_min'], globs['sp_B_sweep_max'], 100)
    #     # By0_G_per_V = 1.8922
    #     # By0_nulling = 0.1504
    #     By0_G_per_V = 2.113
    #     By0_nulling = 0.146
    #     # By0_nulling = 0.170
    #     ByGr = By0_G_per_V*(By0r-By0_nulling)
    #     pkHz = 702.37
    #     fLr = pkHz * np.abs(ByGr)
    #     plt.plot(tr, fLr, c='w')
    #axis(ymin=1e-3*(fsp-frange/2), ymax=1e-3*(fsp+frange/2))
    axis(xmin=t1[0], xmax=t1[-2], ymin=1e-3*subset[0], ymax=1e-3*subset[1])
    xlabel('Time (s)')
    ylabel('Frequency (kHz)')
    title(shot_id)
    if save_png:
        savefig(os.path.join(folder, shot_id+'_stft.png'))
    show()

    if import_ref:
        plt.figure('Reference spectrogram',figsize=(9,4.787))
        plot_stft(res_ref[:-1,], t1_ref[:-1], f1_ref, range=ref_stft_range, png=None)#, cmap='gray_r')
        # axhline(fL_guess/1e3, c='w', ls='--', lw=1)
        # axhline(fsp/1e3, c='m', ls='--', lw=1)
        # if not (field_calibration and estimate_snr and globs['sp_B_sweep']):
        #     axhline(fsp/1e3-2, c='lime', ls='--', lw=1)
        #     axhline(fsp/1e3+2, c='lime', ls='--', lw=1)
        # else:
        #     tr = linspace(t_pre_tip_wait, t_faraday_hold, 100)
        #     By0r = linspace(globs['sp_B_sweep_min'], globs['sp_B_sweep_max'], 100)
        #     # By0_G_per_V = 1.8922
        #     # By0_nulling = 0.1504
        #     By0_G_per_V = 2.113
        #     By0_nulling = 0.146
        #     # By0_nulling = 0.170
        #     ByGr = By0_G_per_V*(By0r-By0_nulling)
        #     pkHz = 702.37
        #     fLr = pkHz * np.abs(ByGr)
        #     plt.plot(tr, fLr, c='w')
        #axis(ymin=1e-3*(fsp-frange/2), ymax=1e-3*(fsp+frange/2))
        axis(xmin=t1_ref[0], xmax=t1_ref[-2], ymin=1e-3*subset[0], ymax=1e-3*subset[1])
        xlabel('Time (s)')
        ylabel('Frequency (kHz)')
        title(shot_id + ' reference')
        show()

    figure('Faraday amplitude (a.u.)')
    plot(t1, res_t)
    if estimate_snr:
        axhline(A_atoms, c='g',  ls='--', lw=1, label='atoms')
        axhline(A_photon, c='r', ls='-.', lw=1, label='photon')
        axhline(A_electronic, c='k', ls=':', lw=1, label='electronic')
        legend(shadow=True, loc='upper right')
        axvline(t_faraday_hold, c='g', ls='--', lw=1)
        axvline(t_pre_tip_wait, c='r', ls='-.', lw=1)
        axvline(t_hobbs_settle, c='k', ls=':', lw=1)
        text(0.9*t1.max(), 0.6*res_t.max(), 'SNR (atoms)= {:.1f}\nSNR (photon)= {:.1f}'.format(SNR_atoms, SNR_photon), {'fontsize': 14, 'horizontalalignment': 'right'})
    xlabel('time (s)')
    title(shot_id)
    savefig(os.path.join(folder, shot_id+'_faraday.png'))
    show()

    import gc
    gc.collect()

    if noise_calibration:
        dt=1/5000000
        t = np.arange(60000*dt,3000000*dt,dt)
        sensor = V[60000:3000000]

        def butter_bandpass(lowcut, highcut, fs, order=3):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                sos = butter(order, [low, high], analog=False, btype='band', output='ba')
                return sos

        def noise(x,c50,s50,c150,s150,c250,s250,c350,s350,const):
            return c50*np.cos(2*np.pi*50*x) + s50*np.sin(2*np.pi*50*x) + c150*np.cos(2*np.pi*150*x) + s150*np.sin(2*np.pi*150*x) + c250*np.cos(2*np.pi*250*x) + s250*np.sin(2*np.pi*250*x) + c350*np.cos(2*np.pi*350*x) + s350*np.sin(2*np.pi*350*x) + const

        b, a = butter_bandpass(842500,847500,5000000)

        filtered_data = filtfilt(b, a, sensor)

        # dt=0.000001
        # t = np.arange(0,0.5,dt)
        # sensor = np.cos(2*np.pi*(1000)*t - 5*np.cos(2*np.pi*10*t))

        fig1 = plt.figure()

        plt.plot(t,filtered_data)
        plt.title("Signal")

        fig2 = plt.figure()

        analytical_signal = hilbert(filtered_data)
        plt.plot(t,analytical_signal.real)
        plt.plot(t,analytical_signal.imag)
        plt.title("Analytical signal")
        plt.legend(["Real","Imaginary"])

        fig3 = plt.figure()

        phase_data = []

        for x in range(len(analytical_signal)):
            phase_data.append(2*(math.atan(analytical_signal.imag[[x]]/analytical_signal.real[[x]])))

        phase_data_unwrapped = np.unwrap(phase_data)/2

        phase_data_unwrapped = phase_data_unwrapped.real

        inst_freq = np.diff(phase_data_unwrapped)/(2*np.pi*dt)

        export_data = False
        if export_data:
            import pandas as pd
            data_DF = pd.DataFrame(inst_freq)
            file_str = 'Z:/Experiments/spinor/crossed_beam_bec/so_simple_exports/farday_free_induction4'
            print("Exporting data to csv file at " + file_str)
            data_DF.to_csv(file_str + '.csv', index=False)
            savefig(file_str + '.png', dpi=300)

        def butter_lowpass(cutoff, fs, order=3):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False,output='ba')
            return b, a

        d, c = butter_lowpass(1000,5000000)

        # inst_freq = filtfilt(d, c, inst_freq)

        popt, pcov = curve_fit(noise, t[15000:215000]-0.015, inst_freq[15000:215000])

        model = lmfit.Model(noise)
        params = model.make_params(c50 = 100, s50 = 100,c150=100,s150=100,c250=100,s250=100,c350=100,s350=100, const = 0)
        result = model.fit(inst_freq[15000:215000], params, x=t[15000:215000]-0.015)
        print(result.fit_report())

        plt.plot(t[15000:-2]-0.015,inst_freq[15000:-1])
        plt.title("Larmor frequency")
        plt.xlim(0,0.045)
        plt.ylim(842000,848000)

        plt.plot(t[0:-1]-0.015, noise(t[0:-1]-0.015, *popt), 'r--')

        plt.show()

        fig4 = plt.figure()

        yf = fft(inst_freq[15000:215000])
        xf = fftfreq(200000, 1 / 5000000)

        yf=yf/(100000)

        plt.plot(xf, np.abs(yf))
        plt.title("FFT of Larmor Frequency")
        plt.xlim(25,1000)
        plt.ylim(0,750)
        plt.show()
        print(0.5*yf[0],yf[2],yf[6],yf[10],yf[14],yf[18])

        phase_data_for_fit = phase_data_unwrapped[15000:415000]

        fig5 = plt.figure()
        plt.plot(np.arange(75000*dt,(len(phase_data_for_fit)+75000)*dt,dt),phase_data_for_fit-2*np.pi*0.5*yf[0]*np.arange(15000*dt,(len(phase_data_for_fit)+15000)*dt,dt))

        def noise_phase(x,c50,s50,c150,s150,c250,s250,c350,s350,const):
            return c50*np.sin(2*np.pi*50*x)/(2*np.pi*50) - s50*np.cos(2*np.pi*50*x)/(2*np.pi*50) + c150*np.sin(2*np.pi*150*x)/(2*np.pi*150) - s150*np.cos(2*np.pi*150*x)/(2*np.pi*150) + c250*np.sin(2*np.pi*250*x)/(2*np.pi*250) - s250*np.cos(2*np.pi*250*x)/(2*np.pi*250) + c350*np.sin(2*np.pi*350*x)/(2*np.pi*350) - s350*np.cos(2*np.pi*350*x)/(2*np.pi*350) + const

        popt, pcov = curve_fit(noise_phase, np.arange(75000*dt,(len(phase_data_for_fit)+75000)*dt,dt)-0.015, phase_data_for_fit-2*np.pi*0.5*yf[0].real*np.arange(15000*dt,(len(phase_data_for_fit)+15000)*dt,dt))

        print(popt/(2*np.pi))