close all; clear variables;

fs = 48000;
L_EDC = 10; % seconds
rtRange = [1, 15];
minA = -45; % dB
noiseAmplitudeRange = [-14, -2];

nEDCs = 100000; % per slope
nMaxSlopes = 3;

%% Generate nEDCs EDCs with nSlopes slopes each

edcs = zeros(nEDCs * nMaxSlopes, 100);
noiseLevels = zeros(nEDCs * nMaxSlopes, 1);
aVals = zeros(nEDCs * nMaxSlopes, 3);

t_ori = (0:round(L_EDC*fs)-1).' / fs;

for nSlopes = 1:nMaxSlopes
    ofb = octaveFilterBank('1 octave', fs, 'FilterOrder', 8);

    for edcIdx = 1:nEDCs
        fprintf('Generating EDC %d/%d with %d slopes.\n', edcIdx, nEDCs, nSlopes);

        % Randomly draw T
        T = sort(rand(nSlopes, 1)*(rtRange(2)-rtRange(1)) + rtRange(1));
        TDiffs = circshift(T, -1) ./ T;
        % Make sure that drawn T values are different enough
        while any(TDiffs(1:nSlopes-1) < 1.5)
            T = sort(rand(nSlopes, 1)*(rtRange(2)-rtRange(1)) + rtRange(1));
            TDiffs = circshift(T, -1) ./ T;
        end

        noiseVal = 10^(rand(1,1)*(noiseAmplitudeRange(2) - noiseAmplitudeRange(1)) + noiseAmplitudeRange(1)); % (nSlopes*min(A)/(L_EDC*fs)) *

        % Randomly draw A in dB
        thisMinA = minA; % pow2db(noiseVal) + 20; % max([minA, pow2db(noiseVal * L_EDC*fs) + 20]);
        A = 10.^((rand(nSlopes, 1)*(0-thisMinA) + thisMinA)/10);
        A = sort(A, 'descend'); % sort descending to get proper bumps
        ADiffs = A ./ circshift(A, -1);

        % Make sure that drawn A values are different enough
        while any(ADiffs(1:nSlopes-1) < 10)
            A = 10.^((rand(nSlopes, 1)*(0-thisMinA) + thisMinA)/10);
            A = sort(A, 'descend'); % sort descending to get proper bumps
            ADiffs = A ./ circshift(A, -1);
        end

        fs_ds = fs / (fs*L_EDC/100);
        t_ds = (0:round(L_EDC*fs_ds)-1).' / fs_ds;

        % normalize to 0dB
        targetSumA = 1 - L_EDC*fs_ds*noiseVal;
        A = targetSumA * A / sum(A);

        % Convert to generating exponential values
        A_gen = 13.81*L_EDC*A./T;
        T_gen = T;
        noiseVal_gen = L_EDC*fs_ds*noiseVal;

        % Generate EDC based on decaying noise
        fBandIdx = randi(6) + 2; % first band in ofb is 30 Hz
        decayingNoiseEDC = generateSyntheticDecayingNoiseEDCFiltered(T_gen, A_gen, noiseVal_gen, t_ori, ofb, fBandIdx);

        % Do backwards integration
        decayingNoiseEDC = schroederInt(decayingNoiseEDC);

        % Normalize to 0dB
        normFactor = max(decayingNoiseEDC);
        decayingNoiseEDC = decayingNoiseEDC / normFactor;
        A = A / normFactor;
        noiseVal = noiseVal / normFactor;
        
        % Add random offset between [-10dB, 10dB]
        offset = 10^(-2*rand+1);
        A = A .* offset;
        noiseVal = noiseVal .* offset;
        decayingNoiseEDC = decayingNoiseEDC .* offset;

        % Discard last 5 percent of samples
        decayingNoiseEDC(round(0.95*length(decayingNoiseEDC))+1:end) = [];

        % Downsampling
        decayingNoiseEDC = resample(decayingNoiseEDC, 100, length(decayingNoiseEDC), 0, 5);

        % Save values
        noiseLevels((nSlopes-1)*nEDCs + edcIdx) = noiseVal;
        aVals((nSlopes-1)*nEDCs + edcIdx, 1:length(A)) = A;

%         reconstructionFromModel = generateSyntheticEDC(T, A, noiseVal, 0, t_ds);
%         figure;
%         plot(t_ds, pow2db(decayingNoiseEDC))
%         hold on
%         plot(t_ds, pow2db(reconstructionFromModel))
%         legend('Decaying Noise EDC', 'Model');
%         disp(immse(pow2db(decayingNoiseEDC(1:0.99*L_EDC*fs_ds)), pow2db(reconstructionFromModel(1:0.99*L_EDC*fs_ds))));

        % Save EDCs
        edcs((nSlopes-1)*nEDCs + edcIdx, 1:length(decayingNoiseEDC)) = decayingNoiseEDC;
    end
end
%% Save all
assert(all(edcs > 0, 'all'), 'Negative EDC values.');

wrkDir = fullfile('..', 'data');
save(fullfile(wrkDir, 'edcs_100.mat'), 'edcs', '-v7.3')
save(fullfile(wrkDir, 'noiseLevels_100.mat'), 'noiseLevels', '-v7.3')
save(fullfile(wrkDir, 'aVals_100.mat'), 'aVals', '-v7.3')
