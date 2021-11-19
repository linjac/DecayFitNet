classdef DecayFitNetToolbox < handle
    
    properties
        sampleRate
        
        % filter frequency = 0 will give a lowpass band below the lowest
        % octave band, filter frequency = sample rate / 2 will give the
        % highpass band above the highest octave band
        filterFrequencies = [125, 250, 500, 1000, 2000, 4000]
    end
    properties (SetAccess = private)
        version = '0.1.0'
        output_size = 100  % Timesteps of downsampled RIRs
        nSlopes
        PATH_ONNX
        onnx_model
        networkName
        input_transform
    end
    
    methods
        function obj = DecayFitNetToolbox(nSlopes, sampleRate)
            if nargin < 1
                nSlopes = 0;
            end
            if nargin < 2
                sampleRate = 48000;
            end
                
            obj.sampleRate = sampleRate;
            obj.nSlopes = nSlopes;
            
            % Load onnx model
            [thisDir, ~, ~] = fileparts(mfilename('fullpath'));
            obj.PATH_ONNX = fullfile(thisDir, '..', 'model');
            
            if obj.nSlopes == 0
                % infer number of slopes with network
                slopeMode = '';
            elseif obj.nSlopes == 1
                % fit exactly 1 slope plus noise
                slopeMode = '1slopes_';
            elseif obj.nSlopes == 2
                % fit exactly 2 slopes plus noise
                slopeMode = '2slopes_';
            elseif obj.nSlopes == 3
                % fit exactly 3 slopes plus noise
                slopeMode = '3slopes_';
            else
                error('Please specify a valid number of slopes to be predicted by the network (nSlopes=1,2,3 for 1,2,3 slopes plus noise, respectively, or nSlopes=0 to let the network infer the number of slopes [max 3 slopes]).');
            end
            obj.networkName = sprintf('DecayFitNet_%s', slopeMode);
            
            % Load ONNX model:
            if exist([obj.networkName, 'model.mat'], 'file')
                fprintf('Loading precompiled model %smodel.mat\n', obj.networkName)
                obj.onnx_model = load([obj.networkName, 'model.mat']).tmp;
            else
                obj.onnx_model = importONNXFunction(fullfile(obj.PATH_ONNX, [obj.networkName, 'v9.onnx']), [obj.networkName, 'model']);
                tmp = obj.onnx_model;
                save([obj.networkName, 'model.mat'], 'tmp');
            end
            [~, msgid] = lastwarn;
            if strcmp(msgid, 'MATLAB:load:cannotInstantiateLoadedVariable')
                error('Could not load the ONNX model. Please make sure you are running MATLAB 2020b or later and installed the following toolbox to load ONNX models in MATLAB: https://se.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format');
            end
            
            disp(obj.onnx_model)

            fid = py.open(fullfile(obj.PATH_ONNX, sprintf('input_transform_%sp2.pkl', slopeMode)),'rb');
            obj.input_transform = py.pickle.load(fid);
        end
        
        function set.filterFrequencies(obj, filterFrequencies)
            assert(~any(filterFrequencies < 0) && ~any(filterFrequencies > obj.sampleRate/2), 'Invalid band frequency. Octave band frequencies must be bigger than 0 and smaller than fs/2. Set frequency=0 for a lowpass band and frequency=fs/2 for a highpass band.');
            filterFrequencies = sort(filterFrequencies);
            obj.filterFrequencies = filterFrequencies;
        end
                
        function [edcs, scaleAdjustFactors] = preprocess(obj, signal)
            nBands = length(obj.filterFrequencies);
            
            nRirs = 1;
            edcs = zeros(obj.output_size, nRirs, nBands);
            tAdjustFactors = zeros(1, nRirs, nBands);
            nAdjustFactors = zeros(1, nRirs, nBands);

            % Extract decays
            schroederDecays = rir2decay(signal, obj.sampleRate, obj.filterFrequencies, true, true, true);
            
            rirIdx = 1;
            for bandIdx = 1:nBands
                % Do backwards integration and remove trailing zeroes
                thisDecay = schroederDecays(:, bandIdx);
                                
                % Convert to dB and clamp at -140dB
                thisDecay = pow2db(thisDecay);
                thisLength = find(thisDecay < -140, 1);
                if ~isempty(thisLength)
                    thisDecay = thisDecay(1:thisLength);
                end
                
                % Calculate adjustment factors for t and n predictions
                tAdjustFactors(:, rirIdx, bandIdx) = 10/(length(thisDecay)/obj.sampleRate);
                nAdjustFactors(:, rirIdx, bandIdx) = length(thisDecay) / 100;
                
                % Discard last 5%
                thisDecay = DecayFitNetToolbox.discardLast5(thisDecay);

                % Resample to obj.output_size (default = 100) samples
                thisDecay_ds = resample(thisDecay, obj.output_size, length(thisDecay), 0, 5);
                edcs(1:obj.output_size, rirIdx, bandIdx) = thisDecay_ds(1:obj.output_size);

                tmp = 2 * edcs(1:obj.output_size, rirIdx, bandIdx) ./ obj.input_transform{'edcs_db_normfactor'};
                edcs(1:obj.output_size, rirIdx, bandIdx) = tmp + 1;
            end
            
            edcs = squeeze(edcs);
            scaleAdjustFactors.tAdjust = squeeze(tAdjustFactors);
            scaleAdjustFactors.nAdjust = squeeze(nAdjustFactors);
        end
        
        function [t_prediction, a_prediction, n_prediction] = estimate_parameters(obj, rir)
            [edcs, scaleAdjustFactors] = obj.preprocess(rir);
            
            % Forward pass of the DecayFitNet
            net = str2func([obj.networkName, 'model']);
            [t_prediction, a_prediction, n_prediction, n_slopes_probabilities] = net(edcs, obj.onnx_model, 'InputDataPermutation', [2,1]);
            
            [t_prediction, a_prediction, n_prediction] = obj.postprocess_parameters(t_prediction, a_prediction, n_prediction, n_slopes_probabilities, true, scaleAdjustFactors);
        end
        
        function [t_prediction, a_prediction, n_prediction, n_slopes_prediction] = postprocess_parameters(obj, t_prediction, a_prediction, n_prediction, n_slopes_probabilities, sort_values, scaleAdjustFactors)
        %% Process the estimated t, a, and n parameters (output of the decayfitnet) to meaningful values
            if ~exist('sort_values', 'var')
                sort_values = true;
            end

            % Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
            n_prediction = min(max(n_prediction, -32), 32);

            % Go from noise exponent to noise value
            n_prediction = 10 .^ n_prediction;

            % In nSlope inference mode: Get a binary mask to only use the number of slopes that were predicted, zero others
            if obj.nSlopes == 0
                [~, n_slopes_prediction] = max(n_slopes_probabilities, [], 1);
                tmp = repmat(linspace(1,3,3), [size(n_slopes_prediction,2), 1])';
                mask = tmp <= repmat(n_slopes_prediction, [3,1]);
                a_prediction(~mask) = 0;
            end

            t_prediction = t_prediction ./ scaleAdjustFactors.tAdjust.';
            n_prediction = n_prediction ./ scaleAdjustFactors.nAdjust.';

           if sort_values
                % Sort T and A values:
                
                % 1) only in nSlope inference mode: assign nans to sort the inactive slopes to the end
                if obj.nSlopes == 0
                    t_prediction(~mask) = NaN;
                    a_prediction(~mask) = NaN;
                end

                % 2) sort
                [t_prediction, sort_idxs] = sort(t_prediction, 1);
                for batchIdx = 1: size(a_prediction, 2)
                    a_this_batch = a_prediction(:, batchIdx);
                    a_prediction(:, batchIdx) = a_this_batch(sort_idxs(:, batchIdx));
                end
                
                % 3) only in nSlope inference mode: set nans to zero again
                if obj.nSlopes == 0
                    t_prediction(isnan(t_prediction)) = 0;  
                    a_prediction(isnan(a_prediction)) = 0; 
                end
           end
        end
    end
    
    methods(Static)    
        function edc = generate_synthetic_edcs(T, A, noiseLevel, t)
            %% Generates an EDC from the estimated parameters.  
            assert(size(T, 2) == size(A, 2) && size(T, 2) == size(noiseLevel, 2), 'Wrong size in the input (different batch size in T, A, N)')
            [nSlopes, batchSize] = size(T);
            nSamples = length(t);
            
            % Permute to [batch_size, n_slopes], for consistency with the toolbox.
            T = permute(T, [2,1]);
            A = permute(A, [2,1]);
            noiseLevel = permute(noiseLevel, [2,1]);
            
            % Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
            tau_vals = -log(1e-6) ./ T;
            
            % Repeat values such that end result will be (sampled_idx, batch_size, n_slopes)
            %t_rep = repmat(t, [1, size(T, 2), size(T, 1)]);
            %tau_vals_rep = repmat(permute(tau_vals, [2,1]), [size(t, 1), 1, 1 ]);
            % Repeat values such that end result will be (batch_size, n_slopes, sampled_idx)
            t_rep = repmat(permute(t, [1, 3, 2]), [batchSize, nSlopes, 1]);
            tau_vals_rep = repmat(permute(tau_vals, [1,2,3]), [1, 1, nSamples]);
            assert(all(size(t_rep) == size(tau_vals_rep)), 'Wrong size in tau')
            
            % Calculate exponentials from decay rates
            time_vals = -t_rep .* tau_vals_rep;
            exponentials = exp(time_vals);
            
            % Zero exponentials where T=A=0
            for batchIdx = 1:batchSize
                zeroT = (T(batchIdx, :) == 0);
                exponentials(batchIdx, zeroT, :) = 0;
            end
            
            % Offset is required to make last value of EDC be correct
            exp_offset = repmat(exponentials(:, :, end), [1, 1, nSamples]);
            
            % Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
            A_rep = repmat(A, [1, 1, nSamples]);
            
            % Multiply exponentials with their amplitudes and sum all exponentials together
            edcs = A_rep .* (exponentials - exp_offset);
            edc = reshape(sum(edcs, 2), batchSize, nSamples);

            % Add noise
            noise = noiseLevel .* linspace(nSamples, 1, nSamples);
            edc = edc + noise;
        end
        
        function output = discardLast5(signal)
            % Discard last 5% samples of a signal
            assert(size(signal,1) > size(signal,2), 'The signal should be in [timesteps, channels]')
            last5 = round(0.95 * size(signal,1));
            
            output = zeros(last5, size(signal,2));
            for channel = 1:size(signal,2)
                tmp = signal(:,channel);
                tmp(last5+1:end) = [];
                output(:,channel) = tmp;
            end
        end
    end
    
end
