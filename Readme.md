This repository contains the original python source files used to process a wav file to remove an unknown periodic sinusoidal interference, used to answer [this question on stackoverflow](https://stackoverflow.com/questions/53324732/matlab-remove-high-frequency-noise-from-wav-file). I am also posting my answer here for completeness.

*Disclaimer*: The answer was provided in Matlab since that was the tag used on the question. However I only had a python environment to perform this processing, so while I tried to transpose it to Matlab's equivalent, there may be cases where things got missed (e.g. based 1 indexing, typos, ...)

---
Given the fairly dynamic nature of the interference in your sample, stationary filters are not going to yield very satisfying results. To improve performance, you would need to dynamically adjust the filtering parameters based on estimates of the interference.

Fortunately in this case the interference is pretty strong and exhibits a fairly regular pattern which makes it easier to estimate. This can be seen from the signal's [`spectrogram`](https://www.mathworks.com/help/signal/ref/spectrogram.html).
For the following derivations we will be assuming the samples of the wavfile has been stored in the array `x` and that the sampling rate is `fs` (which is 8000Hz in the provided sample).

    [Sx,f,t] = spectrogram(x, triang(1024), 1023, [], fs, 'onesided');

[![enter image description here][1]][1]

Given that the interference is strong, obtaining the frequency of the interference can be done by locating the peak frequency in each time slice:

    frequency = zeros(size(Sx,2),1);
    for k = 1:size(Sx,2)
        [pks,loc] = findpeaks(Sx(:,k));
        frequency(k) = fs * (loc(1)-1);
    end

Seeing that the interference is periodic we can use the Discrete Fourier Transform to decompose this signal:

    M = 32*fs;
    Ff = fft(frequency, M);
    plot(fs*[0:M-1]/M, 20*log10(abs(Ff));
    axis(0, 2);
    xlabel('frequency (Hz)');
    ylabel('amplitude (dB)');

[![enter image description here][2]][2]

Using the first two harmonics as an approximation, we can model the frequency of the interference signal as:

    T = 1.0/fs
    t = [0:length(x)-1]*T
    freq = 750.0127340203496
           + 249.99913423501602*cos(2*pi*0.25*t - 1.5702946346796276)
           + 250.23974282864816*cos(2*pi*0.5 *t - 1.5701043282285363);

At this point we would have enough to create a narrowband filter with a center frequency (which would change dynamically as we keep updating the filter coefficients) given by that frequency model. Note however that constantly recomputing and updating the filter coefficient is a fairly expensive process and given that the interference is strong, it is possible to do better by locking on to the interference phase. This can be done by correlating small blocks of the original signal with sine and cosine at desired frequency. We can then slightly tweak the phase to align the sine/cosine with the original signal.

    % Compute the phase of the sine/cosine to correlate the signal with
    delta_phi = 2*pi*freq/fs;
    phi = cumsum(delta_phi);

    % We scale the phase adjustments with a triangular window to try to reduce
    % phase discontinuities. I've chosen a window of ~200 samples somewhat arbitrarily,
    % but it is large enough to cover 8 cycles of the interference around its lowest
    % frequency sections (so we can get a better estimate by averaging out other signal
    % contributions over multiple interference cycles), and is small enough to capture
    % local phase variations.
    step = 50;
    L    = 200;
    win  = triang(L);
    win  = win/sum(win);

    for i = 0:floor((length(x)-(L-step))/step)
        % The phase tweak to align the sine/cosine isn't linear, so we run a few
        % iterations to let it converge to a phase locked to the original signal
        for iter = 0:1
            xseg   = x[(i*step+1):(i*step+L+1)];
            phiseg = phi[(i*step+1):(i*step+L+1)];
            r1 = sum(xseg .* cos(phiseg));
            r2 = sum(xseg .* sin(phiseg));
            theta = atan2(r2, r1);

            delta_phi[(i*step+1):(i*step+L+1)] = delta_phi[(i*step+1):(i*step+L+1)] - theta*win;
            phi = cumsum(delta_phi);
        end
    end

Finally, we need to estimate the amplitude of the interference. Here we choose to perform the estimation over the initial 0.15 seconds where there is a little pause before the speech starts so that the estimation is not biased by the speech's amplitude:

    tmax = 0.15;
    nmax = floor(tmax * fs);
    amp  = sqrt(2*mean(x[1:nmax].^2));
    % this should give us amp ~ 0.250996990794946

These parameters then allow us to fairly precisely reconstruct the interference, and correspondingly remove the interference from the original signal by direct subtraction:

    y = amp * cos(phi)
    x = x-y

[![enter image description here][3]][3]

Listening to the resulting output, you may notice a remaining faint whooshing noise, but nothing compared to the original interference. Obviously this is a fairly ideal case where the parameters of the interference are so easy to estimate that the results almost look too good to be true. You may not get the same performance with more random interference patterns.

*Note*: python script used for this processing can be found [here](https://github.com/SleuthEye/SO53324732).

  [1]: https://i.stack.imgur.com/S82It.png
  [2]: https://i.stack.imgur.com/t1K1C.png
  [3]: https://i.stack.imgur.com/z7Uem.png
