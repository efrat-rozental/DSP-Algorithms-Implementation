clear; clc; close all;

%% PART A: Decimation in Time FFT / IFFT
disp('--- PART A: Testing Custom FFT/IFFT ---');
% Create a random signal padded to a power of 2 for testing
N_test = 1024; 
x_test = randn(N_test, 1) + 1j * randn(N_test, 1);

% Compute using custom functions
X_iter = my_fft_iterative(x_test);
X_rec  = my_fft_recursive(x_test);
X_matlab = fft(x_test);

x_ifft_iter = my_ifft_iterative(X_iter);
x_ifft_rec  = my_ifft_recursive(X_rec);

% Verify accuracy against MATLAB's built-in functions
err_fft_iter = max(abs(X_iter - X_matlab));
err_fft_rec = max(abs(X_rec - X_matlab));
err_ifft_iter = max(abs(x_ifft_iter - x_test));
err_ifft_rec = max(abs(x_ifft_rec - x_test));

fprintf('Max Error Iterative FFT: %e\n', err_fft_iter);
fprintf('Max Error Recursive FFT: %e\n', err_fft_rec);
fprintf('Max Error Iterative IFFT: %e\n', err_ifft_iter);
fprintf('Max Error Recursive IFFT: %e\n\n', err_ifft_rec);

%% ========================================================================
%% PART B: Digital Filtering
disp('--- PART B: Digital Filtering ---');
% 1. Load Filter 
% (Using dynamic field extraction to bypass unknown variable names)
temp = load('filter_0.25_101.mat');
fields = fieldnames(temp);
h_b = temp.(fields{1}); 

% 2. Setup parameters based on calculations
Fs_B = 10; % Calculated in analytical part
N_sig = 1947; % Needed to get 2048 output samples

% 3. Generate Signal r(t)
t = (0:N_sig-1)' / Fs_B;
r_n = cos(2*pi*1*t) + cos(2*pi*2.5*t);

% 4. Filter signal using custom convolution
S_n = my_linear_conv(r_n, h_b); % Result is 2048 samples

% 5. Compute DFT and Plot
S_k = my_fft_iterative(S_n);
S_k_mag = abs(S_k);

% Frequency axis mapping [0, Fs]
f_axis = (0:2047)' * (Fs_B / 2048);

figure('Name', 'Part B: Filtered Signal Spectrum');
plot(f_axis, S_k_mag, 'LineWidth', 1.5);
title('Magnitude Spectrum of Filtered Output |S[k]|');
xlabel('Analog Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs_B]); grid on;

%% ========================================================================
%% PART C: Overlap-Save (OVS) Convolution
disp('--- PART C: Overlap-Save Convolution ---');

% Load data
temp = load('sig_x.mat'); f_x = fieldnames(temp); x_c = temp.(f_x{1});
temp = load('filter_1.mat'); f_h1 = fieldnames(temp); h1 = temp.(f_h1{1});
temp = load('filter_2.mat'); f_h2 = fieldnames(temp); h2 = temp.(f_h2{1});

% Ensure loaded variables are strictly column vectors
x_c = x_c(:);
h1 = h1(:);
h2 = h2(:);

Fs_C = 18000;

% --- Filter Visualization (Question b) ---
figure('Name', 'Part C: Filter Characteristics');

% Filter 1 Time Domain
subplot(2,2,1);
stem(h1, 'Marker', 'none');
title('Filter 1: Impulse Response');
xlabel('n'); ylabel('Amplitude');
grid on;

% Filter 2 Time Domain
subplot(2,2,2);
stem(h2, 'Marker', 'none');
title('Filter 2: Impulse Response');
xlabel('n'); ylabel('Amplitude');
grid on;

% Filter 1 Frequency Domain
subplot(2,2,3);
N_fft_h = 1024; % Zero-padding for a smoother frequency response
H1_k = my_fft_iterative([h1; zeros(N_fft_h - length(h1), 1)]);
f_axis_h = (0:floor(N_fft_h/2)) * (Fs_C / N_fft_h);
plot(f_axis_h, abs(H1_k(1:floor(N_fft_h/2)+1)));
title('Filter 1: Magnitude Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
grid on;

% Filter 2 Frequency Domain
subplot(2,2,4);
H2_k = my_fft_iterative([h2; zeros(N_fft_h - length(h2), 1)]);
plot(f_axis_h, abs(H2_k(1:floor(N_fft_h/2)+1)));
title('Filter 2: Magnitude Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
grid on;
% -----------------------------------------------

% --- Signal Analysis for Part C (Question a) ---
figure('Name', 'Part C: Input Signal x[n] Analysis');

% Time Domain Plot
subplot(2,1,1);
t_c = (0:length(x_c)-1) / Fs_C;
plot(t_c, x_c);
title('Input Signal x[n] - Time Domain');
xlabel('Time (Seconds)'); ylabel('Amplitude');
grid on;

% Frequency Domain Plot
subplot(2,1,2);
X_c_fft = my_fft_iterative(x_c);
N_xc = length(x_c);
f_axis_c = (0:floor(N_xc/2)) * (Fs_C / N_xc);
plot(f_axis_c, abs(X_c_fft(1:floor(N_xc/2)+1)));
title('Magnitude Spectrum of x[n]');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
grid on;
% -----------------------------------------------

% --- Question C (): Direct Convolution ---
tic;
y1_dir = my_linear_conv(x_c, h1);
time_dir_h1 = toc;

tic;
y2_dir = my_linear_conv(x_c, h2);
time_dir_h2 = toc;

fprintf('Direct Convolution Time (Filter 1): %.4f seconds\n', time_dir_h1);
fprintf('Direct Convolution Time (Filter 2): %.4f seconds\n', time_dir_h2);

% --- Question D & E (ד, ה): OVS Performance ---
% Test OVS with varying Frame Sizes (N)
% Note: N must be > length(h)-1. Usually chosen as power of 2.
M = length(h1);
pow2_start = nextpow2(M) + 1;
N_vals = 2.^(pow2_start : pow2_start + 5); 

times_ovs = zeros(length(N_vals), 1);

for i = 1:length(N_vals)
    N = N_vals(i);
    tic;
    y1_ovs = my_ovs(x_c, h1, N);
    times_ovs(i) = toc;
end

[min_time, best_idx] = min(times_ovs);
best_N = N_vals(best_idx);
fprintf('Optimal OVS Frame Size (N): %d\n', best_N);
fprintf('Optimal OVS Time: %.4f seconds\n', min_time);

% Calculate OVS for h2 using best N
y2_ovs = my_ovs(x_c, h2, best_N);

% Plot Performance Comparison
figure('Name', 'Part C: Performance OVS vs Direct');
plot(N_vals, times_ovs, '-o', 'LineWidth', 2); hold on;
yline(time_dir_h1, 'r--', 'LineWidth', 2);
title('Execution Time: Direct Convolution vs Overlap-Save');
xlabel('Frame Size (N)');
ylabel('Time (Seconds)');
legend('OVS Method', 'Direct Method');
grid on;

% --- Question F (ו): Output Verification ---
figure('Name', 'Part C (ו): Output Verification');
subplot(2,1,1);
plot(y1_dir, 'b'); hold on; plot(y1_ovs, 'r--');
title('Output Comparison: Filter 1');
legend('Direct', 'OVS'); grid on; xlim([0 5000]);

subplot(2,1,2);
plot(y2_dir, 'b'); hold on; plot(y2_ovs, 'r--');
title('Output Comparison: Filter 2');
legend('Direct', 'OVS'); grid on; xlim([0 5000]);


%% ========================================================================
%% LOCAL FUNCTIONS 

function X = my_fft_recursive(x)
    % Recursive Radix-2 DIT FFT
    x = x(:);
    N = length(x);
    if N <= 1
        X = x;
        return;
    end
    
    % Ensure power of 2 padding if necessary
    if bitand(N, N-1) ~= 0
        N = 2^nextpow2(N);
        x = [x; zeros(N - length(x), 1)];
    end
    
    X_even = my_fft_recursive(x(1:2:end));
    X_odd  = my_fft_recursive(x(2:2:end));
    
    W = exp(-1j * 2 * pi * (0:(N/2 - 1))' / N);
    
    X = [X_even + W .* X_odd; X_even - W .* X_odd];
end

function X = my_fft_iterative(x)
    % Iterative In-Place Radix-2 DIT FFT
    x = x(:);
    N = length(x);
    
    % Pad to pow 2
    bits = nextpow2(N);
    N_pow2 = 2^bits;
    x = [x; zeros(N_pow2 - N, 1)];
    
    % Bit-reversal sorting
    indices = 0:(N_pow2-1);
    rev_indices = bin2dec(fliplr(dec2bin(indices, bits)));
    X = x(rev_indices + 1);
    
    % Butterfly stages (Vectorized)
    for s = 1:bits
        m = 2^s;
        half_m = m / 2;
        
        % Generate twiddle factors for one group and repeat for all groups
        w = exp(-1j * 2 * pi * (0:half_m-1)' / m);
        W = repmat(w, N_pow2 / m, 1);
        
        % Generate indices for top and bottom branches using broadcasting
        top_idx = (1:half_m)' + (0 : m : N_pow2-1);
        top_idx = top_idx(:); 
        bot_idx = top_idx + half_m;
        
        % Execute all butterfly calculations for this stage simultaneously
        t = W .* X(bot_idx);
        u = X(top_idx);
        
        X(top_idx) = u + t;
        X(bot_idx) = u - t;
    end
end

function x = my_ifft_recursive(X)
    % IFFT using the identity: IFFT(X) = (FFT(X*))* / N
    N = length(X);
    x = conj(my_fft_recursive(conj(X))) / N;
end

function x = my_ifft_iterative(X)
    % IFFT using the identity: IFFT(X) = (FFT(X*))* / N
    N = length(X);
    x = conj(my_fft_iterative(conj(X))) / N;
end

function y = my_linear_conv(x, h)
    % Direct Linear Convolution (Vectorized outer-product accumulation)
    x = x(:); h = h(:);
    Nx = length(x); Nh = length(h);
    y = zeros(Nx + Nh - 1, 1);
    
    for i = 1:Nh
        y(i : i + Nx - 1) = y(i : i + Nx - 1) + h(i) * x;
    end
end

function y = my_ovs(x, h, N)
    % Overlap-Save Method for fast linear convolution
    x = x(:); h = h(:);
    Nx = length(x);
    M = length(h);
    L = N - M + 1; % Number of valid output points per frame
    
    if L <= 0
        error('Frame size N must be strictly greater than filter length M-1');
    end
    
    % Pad filter to length N and compute FFT
    h_padded = [h; zeros(N - M, 1)];
    H_k = fft(h_padded);
    
    % Calculate blocks needed for the FULL convolution length, not just Nx
    total_len = Nx + M - 1;
    num_blocks = ceil(total_len / L);
    y = zeros(num_blocks * L, 1);
    
    % Ensure x_padded is long enough to feed all blocks
    required_x_len = (num_blocks - 1) * L + N;
    x_padded = [zeros(M - 1, 1); x];
    if length(x_padded) < required_x_len
        x_padded = [x_padded; zeros(required_x_len - length(x_padded), 1)];
    end
    
    % Process blocks
    for k = 1:num_blocks
        start_idx = (k-1)*L + 1;
        end_idx = start_idx + N - 1;
        
        block = x_padded(start_idx : end_idx);
        
        % Circular convolution via FFT
        BLOCK_k = fft(block);
        Y_k = BLOCK_k .* H_k;
        y_circ = real(ifft(Y_k));
        
        % Discard first M-1 points, save the valid L points
        y((k-1)*L + 1 : k*L) = y_circ(M:end);
    end
    
    % Trim to exact linear convolution length
    y = y(1 : total_len);
end