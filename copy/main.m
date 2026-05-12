%% 5G OFDM MU-MIMO 시뮬레이션 (64-QAM + IQ Imbalance + 현실적 난이도 적용)
clear; clc; close all;

%% 1. 시스템 파라미터 및 안테나 설정
fft_len   = 64;               
mod_type  = 6;                % 64-QAM (비트/심볼 = 6) - 난이도 하향 조정
num_users = 2;                
N_s       = 1;                

% 64-QAM에 적합한 테스트 SNR 대역 (10~30dB)
snr_range = 10:2:30;          
path      = 7;                
iter      = 8000;             % 64-QAM 훈련을 위한 충분한 반복 횟수

tx_ant_config = [8 1 0.5 0.5]; 
rx_ant_config = [4 1 0.5 0.5];
N_tx = tx_ant_config(1) * tx_ant_config(2);
N_rx = rx_ant_config(1) * rx_ant_config(2);
cp_len = fft_len / 4;
data_len = fft_len * mod_type; % 64 * 6 = 384 비트

model = SCM();
model.n_path = path;
model.ant(rx_ant_config(1), tx_ant_config(1)); 

%% 2. 딥러닝 수신기 학습용 데이터 수집 (Phase 1)
disp('--- 딥러닝 모델 학습용 데이터 생성 중 (64-QAM + IQ 결함) ---');
XTrain = [];
YTrain = [];

% 64-QAM의 점들을 구분하기 위한 가중치 벡터 (Bit -> Class 변환용)
bit_weights = 2.^(mod_type-1:-1:0); 

for i = 1:iter
    H = zeros(path, fft_len+cp_len, N_rx * num_users, N_tx);
    Wt = zeros(N_tx, num_users);
    Wr = zeros(N_rx, num_users);
    
    for d = 1:num_users
        [temp_H, rx_angle] = model.FD_channel(fft_len + cp_len);
        H(:,:, 1+(d-1)*N_rx : d*N_rx, :) = temp_H;
        [~, max_idx] = max(abs(temp_H(:,1,1,1)));
        sel_angle = rx_angle(:, max_idx);
        Wt(:, d) = steer_precoding(model.fc, tx_ant_config, sel_angle(1:2));
        Wr(:, d) = steer_precoding(model.fc, rx_ant_config, sel_angle(3:4), 2);
    end
    
    t_He = zeros(path, num_users, num_users);
    for k = 1:path
        tmp_H = squeeze(H(k,1,:,:));
        for d = 1:num_users
            rx_idx = 1+(d-1)*N_rx : d*N_rx;
            t_He(k, d, :) = Wr(:,d).' * tmp_H(rx_idx, :) * Wt;
        end
    end
    He_freq = fft(t_He, fft_len, 1); 
    
    % 현실적인 채널 오류 (256-QAM 보다는 덜 민감하므로 0.005 유지)
    error_variance = 0.005; 
    He_error = sqrt(error_variance/2) * (randn(size(He_freq)) + 1i*randn(size(He_freq)));
    He_est = He_freq + He_error; 
    
    bit_data = randi([0 1], num_users, data_len);
    sym_data = base_mod(bit_data, mod_type); 
    
    [Dsym, ~, Wd] = ZF_precoding(sym_data, He_est); 
    Isym = ifft(Dsym, fft_len, 2) * sqrt(fft_len);
    tx_ofdm = [Isym(:, fft_len - cp_len + 1 : end), Isym];
    tx_signal = Wt * tx_ofdm;
    
    % 폭넓은 SNR 학습 (64-QAM에 맞춰 조정)
    train_snr = randi([15, 35]); 
    [rx_signal, ~] = awgn_noise(model.FD_fading(tx_signal, H), train_snr);
    
    for d = 1:num_users
        rx_idx = 1+(d-1)*N_rx : d*N_rx;
        user_rx = Wr(:,d).' * rx_signal(rx_idx, :);
        user_Isym = user_rx(:, cp_len + 1 : end);
        user_Dsym = fft(user_Isym, fft_len, 2) / sqrt(fft_len);
        
        % 수신단 I/Q 불균형 (하드웨어 결함)
        I_part = real(user_Dsym);
        Q_part = imag(user_Dsym);
        user_Dsym = (1.2 * I_part) + 1i * (0.8 * Q_part * cos(pi/12) - I_part * sin(pi/12));
        
        for k = 1:fft_len
            y_k = user_Dsym(k);
            h_k = squeeze(He_est(k, d, d)); 
            
            features = [real(y_k); imag(y_k); real(h_k); imag(h_k)];
            XTrain = [XTrain, features];
            
            % 6개 비트를 1~64 번호표(Class)로 자동 변환
            sym_bits = bit_data(d, (k-1)*mod_type + 1 : k*mod_type);
            class_idx = sum(sym_bits .* bit_weights) + 1; 
            
            YTrain = [YTrain; categorical(class_idx)];
        end
    end
end
XTrain = XTrain'; 

%% 3. 딥러닝 네트워크 구축 및 학습 (Phase 2)
disp('--- 딥러닝 모델 학습 시작 (64 Classes) ---');
inputSize = 4;
numClasses = 64; % 64-QAM에 맞게 클래스 수 축소

layers = [
    featureInputLayer(inputSize, 'Normalization', 'zscore') 
    fullyConnectedLayer(512) % 뇌 용량은 유지 (더 빠르고 정확하게 학습)
    reluLayer
    fullyConnectedLayer(256) 
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 1024, ...
    'InitialLearnRate', 0.01, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

%% 4. 성능 평가 시뮬레이션 (Phase 4)
disp('--- 실전 진검승부 시작 (64-QAM) ---');
ber_Basic = zeros(1, length(snr_range));
ber_MMSE = zeros(1, length(snr_range)); 
ber_DL = zeros(1, length(snr_range));

test_iter = 300; 

for s_idx = 1:length(snr_range)
    current_snr = snr_range(s_idx);
    err_Basic = 0; err_MMSE = 0; err_DL = 0;
    total_bits = 0;
    noise_var = 10^(-current_snr/10); 
    
    for i = 1:test_iter
        H = zeros(path, fft_len+cp_len, N_rx * num_users, N_tx);
        Wt = zeros(N_tx, num_users);
        Wr = zeros(N_rx, num_users);
        
        for d = 1:num_users
            [temp_H, rx_angle] = model.FD_channel(fft_len + cp_len);
            H(:,:, 1+(d-1)*N_rx : d*N_rx, :) = temp_H;
            [~, max_idx] = max(abs(temp_H(:,1,1,1)));
            sel_angle = rx_angle(:, max_idx);
            Wt(:, d) = steer_precoding(model.fc, tx_ant_config, sel_angle(1:2));
            Wr(:, d) = steer_precoding(model.fc, rx_ant_config, sel_angle(3:4), 2);
        end
        
        t_He = zeros(path, num_users, num_users);
        for k = 1:path
            tmp_H = squeeze(H(k,1,:,:));
            for d = 1:num_users
                rx_idx = 1+(d-1)*N_rx : d*N_rx;
                t_He(k, d, :) = Wr(:,d).' * tmp_H(rx_idx, :) * Wt;
            end
        end
        He_freq = fft(t_He, fft_len, 1);
        
        % Phase 1과 동일한 채널 오류 적용
        error_variance = 0.005; 
        He_error = sqrt(error_variance/2) * (randn(size(He_freq)) + 1i*randn(size(He_freq)));
        He_est = He_freq + He_error; 
        
        bit_data = randi([0 1], num_users, data_len);
        sym_data = base_mod(bit_data, mod_type);
        
        [Dsym, ~, Wd] = ZF_precoding(sym_data, He_est); 
        Isym = ifft(Dsym, fft_len, 2) * sqrt(fft_len);
        tx_ofdm = [Isym(:, fft_len - cp_len + 1 : end), Isym];
        tx_signal = Wt * tx_ofdm;
        
        [rx_signal, ~] = awgn_noise(model.FD_fading(tx_signal, H), current_snr);
        
        for d = 1:num_users
            rx_idx = 1+(d-1)*N_rx : d*N_rx;
            user_rx = Wr(:,d).' * rx_signal(rx_idx, :);
            user_Isym = user_rx(:, cp_len + 1 : end);
            user_Dsym = fft(user_Isym, fft_len, 2) / sqrt(fft_len);
            
            % 실전 테스트 IQ 불균형 적용
            I_part = real(user_Dsym);
            Q_part = imag(user_Dsym);
            user_Dsym = (1.2 * I_part) + 1i * (0.8 * Q_part * cos(pi/12) - I_part * sin(pi/12));
            
            orig_bits = bit_data(d, :); 
            
            % --- [1] Basic Rx ---
            rx_bit_Basic = base_demod(user_Dsym, mod_type);
            err_Basic = err_Basic + sum(orig_bits ~= rx_bit_Basic);
            
            % --- [2] MMSE Rx ---
            rx_Dsym_MMSE = zeros(1, fft_len);
            for k = 1:fft_len
                H_est_k = squeeze(He_est(k, :, :)); 
                Wd_k = squeeze(Wd(k, :, :));
                G_k = H_est_k * Wd_k; 
                
                g_kk = G_k(d, d);
                interf_var = sum(abs(G_k(d, :)).^2) - abs(g_kk)^2;
                total_noise_var = interf_var + noise_var;
                
                w_mmse = conj(g_kk) / (abs(g_kk)^2 + total_noise_var);
                rx_Dsym_MMSE(k) = user_Dsym(k) * w_mmse;
            end
            rx_bit_MMSE = base_demod(rx_Dsym_MMSE, mod_type);
            err_MMSE = err_MMSE + sum(orig_bits ~= rx_bit_MMSE);
            
            % --- [3] Deep Learning Rx (직접 디코딩) ---
            h_k_all = squeeze(He_est(:, d, d)).'; 
            test_features = [real(user_Dsym).', imag(user_Dsym).', real(h_k_all).', imag(h_k_all).'];
            
            pred_classes = classify(net, test_features);
            pred_classes = str2double(string(pred_classes)); 
            
            rx_bit_DL = zeros(1, data_len);
            for k = 1:fft_len
                c_idx = pred_classes(k) - 1;
                dl_bits = zeros(1, mod_type);
                for b = 1:mod_type
                    dl_bits(mod_type - b + 1) = mod(c_idx, 2);
                    c_idx = floor(c_idx / 2);
                end
                rx_bit_DL((k-1)*mod_type + 1 : k*mod_type) = dl_bits;
            end
            err_DL = err_DL + sum(orig_bits ~= rx_bit_DL);
            
            total_bits = total_bits + data_len;
        end
    end
    
    ber_Basic(s_idx) = err_Basic / total_bits;
    ber_MMSE(s_idx) = err_MMSE / total_bits;
    ber_DL(s_idx) = err_DL / total_bits;
    
    fprintf('SNR: %2d dB 완료 (Basic: %d, MMSE: %d, DL: %d)\n', current_snr, err_Basic, err_MMSE, err_DL);
end

%% 5. 결과 시각화
figure;
semilogy(snr_range, ber_Basic, '-bs', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogy(snr_range, ber_MMSE, '-gd', 'LineWidth', 2, 'MarkerSize', 8); 
semilogy(snr_range, ber_DL, '-ro', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)'); ylabel('BER (log scale)');
title('64-QAM BER vs SNR (IQ Imbalance + Fair Channel Error)');
legend('Basic Receiver', 'MMSE Receiver', 'Deep Learning Receiver');
ylim([1e-4 1]);