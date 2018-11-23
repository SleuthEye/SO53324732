import wave
import struct
from scipy.fftpack import fft
from scipy.signal import stft, find_peaks, triang
import numpy as np

def wav_to_floats(wave_file):
    '''
    Read wav file as float array.
    
    This function by yeeking (https://stackoverflow.com/users/1240660/yeeking)
    has been taken from https://stackoverflow.com/a/29989593/2994596
    under CC-BY-SA license.
    '''
    w = wave.open(wave_file)
    astr = w.readframes(w.getnframes())
    # convert binary chunks to short 
    a = struct.unpack("%ih" % (w.getnframes()* w.getnchannels()), astr)
    a = [float(val) / pow(2, 15) for val in a]
    return a, w.getframerate()

def model_interference_frequency(y, fs):
    '''
    Generate a model of the periodic interference using frequency decomposition.
    '''
    N = 16*16000
    m = np.mean(y)
    z = y-m
    A     = [0, 0]
    phase = [0, 0]
    t     = np.arange(0, len(z))/fs

    # The FFT output is affected by the limited length of the input data.
    # As a result we get a better estimate by performing a few iterations
    # were we gradually remove the leftovers from inaccurate estimates.
    for iter in range(0, 3):
        Zf = fft(z, N)
        Zf = Zf[0:len(Zf)//2+1]
        freq = np.arange(0, len(Zf))*fs/N

        # Select the first 2 peaks
        Func  = 20*np.log10(np.abs(Zf))
        Fmax = np.max(Func)
        Thmax = Fmax
        Thmin = np.max([Fmax-70, np.min(Func)])
        while True:
            threshold = 0.5*(Thmin + Thmax)
            peaks,prop = find_peaks(Func, threshold)
            if len(peaks)>2:
                Thmin = threshold
            elif len(peaks)<2:
                Thmax = threshold
            else:
                break
        peaks,prop = find_peaks(Func, threshold)

        # Get the first 2 harmonics
        for i in range(0,len(peaks)):
            H = Zf[peaks[i]]/len(y)
            mag = 2*np.abs(H)
            ang = np.angle(H)
            U = A[i]*np.cos(phase[i]) + mag*np.cos(ang)
            V = A[i]*np.sin(phase[i]) + mag*np.sin(ang)
            A[i]     = np.sqrt(U*U+V*V)
            phase[i] = -np.arccos(U/A[i])

            z -= mag*np.cos(2*np.pi*freq[peaks[i]]*t + ang)
        mu = np.mean(z)
        z -= mu
        m += mu
        
    return [m, A, phase]

def estimate_interference_amplitude(x):
    '''
    Estimate a sinusoidal signal's peak amplitude.
    '''
    return np.sqrt(2*np.mean(x**2))


def interference_frequency_model_eval(m, A, phase, t):
    '''
    Compute the interference's frequency based on estimated model
    parameters (see model_interference_frequency).
    '''
    freq = m + A[0]*np.cos(2*np.pi*0.25*t + phase[0]) \
             + A[1]*np.cos(2*np.pi*0.5 *t + phase[1])
    return freq

def phase_lock(x, freq, fs):
    '''
    Compute the interference's phase by locking onto the strong interference
    from the original signal.
    '''

    # Compute the phase of the sine/cosine to correlate the signal with
    delta_phi = 2*np.pi*freq/fs
    phi = np.cumsum(delta_phi)
    
    # We scale the phase adjustments with a triangular window to try to reduce
    # phase discontinuities. I've chosen a window of ~200 samples somewhat arbitrarily,
    # but it is large enough to cover 8 cycles of the interference around its lowest
    # frequency sections (so we can get a better estimate by averaging out other signal
    # contributions over multiple interference cycles), and is small enough to capture
    # local phase variations.
    step = 50
    L    = 200
    win  = triang(L)
    win /= np.sum(win)
    for i in range(0, (len(x)-(L-step))//step):
        xseg = x[i*step:i*step+L]
        # The phase tweak to align the sine/cosine isn't linear, so we run a few
        # iterations to let it converge to a phase locked to the original signal
        for it in range(0,2):
            phiseg = phi[i*step:i*step+L]
            r1 = np.sum(np.multiply(xseg, np.cos(phiseg)))
            r2 = np.sum(np.multiply(xseg, np.sin(phiseg)))
            theta = np.arctan2(r2, r1)

            delta_phi[i*step:i*step+L] -= theta*win
            phi = np.cumsum(delta_phi)

    return phi

# Load input .wav file with interference
x,fs = wav_to_floats("wav.wav")

n = len(x)
x = np.asarray(x)

# Extract interference frequency
M = 1024
freq_axis,time_axis,z = stft(x, fs, nperseg=M, noverlap=M-1)
pkloc = np.argmax(np.abs(z), 0)

# Compute the parameter of a model for the interference frequency
m,A,phase = model_interference_frequency(freq_axis[pkloc], fs)

T = 1.0/fs
t = np.arange(0,n)*T

# Evaluate the interference frequency at each sampling time
freq = interference_frequency_model_eval(m, A, phase, t)

# Estimate phase and amplitude of the interference
phi  = phase_lock(x, freq, fs)
tmax = 0.15
nmax = int(tmax * fs)
amp  = estimate_interference_amplitude(x[0:nmax])

# Construct estimated interference
y = amp*np.cos(phi)
# Subtract estimated interference from original signal
z = x-y

# Save cleaned result as a .wav file
w = wave.open('filtered.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(2*fs)
w.setnframes(len(z))
z = (z*32767) / np.max(np.abs(z))
w.writeframes(z.astype(int))
w.close()

