function outBands = octaveFiltering(inputSignal, fs, fBands)
numBands = numel(fBands);
outBands = zeros(length(inputSignal), numBands);

for bIdx = 1:numBands
    % Determine IIR filter coefficients for this band
    if fBands(bIdx) == 0
        % Lowpass band below lowest octave band
        fCutoff = (1/sqrt(1.5))*fBands(bIdx+1);
        [z, p, k] = butter(5, fCutoff/fs*2);
    elseif fBands(bIdx) == fs/2
        % Highpass band below lowest octave band
        fCutoff = sqrt(1.5)*fBands(bIdx-1);
        [z, p, k] = butter(5, fCutoff/fs*2, 'high');
    else
        thisBand = fBands(bIdx) .* [1/sqrt(1.5), sqrt(1.5)];
        [z, p, k] = butter(5, thisBand/fs*2, 'bandpass');
    end
    
    % Zero phase filtering
    sos = zp2sos(z, p, k);
    outBands(:, bIdx) = sosfilt(sos, inputSignal);
end

end