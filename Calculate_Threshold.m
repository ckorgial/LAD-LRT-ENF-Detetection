%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program calculates the Mean and Variance values for 
%
% 3. LS-LRT - State of the Art Threshold
%
% ENF detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear;close all;

%%%% Bandpass Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F2  = [0 0.4 0.499 0.4995 0.5 0.5005 0.501 0.6 0.8 1];
M2  = [0 0 0 0.2 1 0.2 0 0 0 0];
BPF = fir2(1023,F2,M2);
BPFF     = abs(fft(BPF,8192));
scalar   = max(BPFF);
BPF      = BPF/scalar;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs       = 400;
T        = 1/fs;
NFFT     = 200*fs;

duration = 5:10;

path     = '';
H0_index = dir(strcat(path,'H0'));

for i = 1:(length(H0_index)-2)
    disp(['i= ',num2str(i)]);
    [audio, fs0] = audioread(strcat(H0_index(i+2).folder,'\',H0_index(i+2).name));
    audio        = audio(:,1)';
    
    for j = 1:length(duration)
        
        current_dur  = duration(j);
        start_index  = randi(length(audio)-current_dur*fs0);
        audio_cut    = audio(start_index:(start_index+current_dur*fs0-1));
        x            = resample(audio_cut, fs, fs0);
        N            = length(x);
        
        x_filtered   = filter(BPF,1,x);
        
        NFFT_full    = max(2^18,2^(nextpow2(N)+2));
        X_filtered   = abs(fft(x_filtered,NFFT_full));
        X_filtered   = X_filtered(1:(end/2+1));
        
        fc           = find(X_filtered==max(X_filtered))*(fs/NFFT_full);
        
        Hc                   = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
        Test_Statistic3(i,j) = 2/N*(x_filtered*Hc)*(Hc'*x_filtered')/((norm(x_filtered).^2));

    end
end
% Save manually both mean30 and var30 as Threshold_info_5_1_10
mean30=mean(Test_Statistic3,1);
var30=var(Test_Statistic3,1);