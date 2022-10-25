%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program calculates the 
%
% 1. LAD-LRT - Laplace Periodogram I
% 2. LS-LRT  - Laplace Periodogram II
% 3. LS-LRT  - Ordinary Peiodogram
%
% for the data under $\mathcal{H}_0 folder$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear;close all;

%%% Bandpass Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F2 = [0 0.4 0.499 0.4995 0.5 0.5005 0.501 0.6 0.8 1];
M2 = [0 0 0 0.2 1 0.2 0 0 0 0];
BPF= fir2(1023,F2,M2);
BPFF     = abs(fft(BPF,8192));
scalar   = max(BPFF);
BPF      = BPF/scalar;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs              = 400;
T               = 1/fs;
AWindowLength   = 16*fs;
AWindowShift    = rectwin(AWindowLength)';
AStepSize       = 1*fs;
NFFT            = 200*fs;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
duration = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path         = '';
H0_index     = dir(strcat(path,'H0'));
H1_index     = dir(strcat(path,'H1'));
%i=10;  % H1
%[audio, fs0] = audioread(strcat(H1_index(i+2).folder,'\',H1_index(i+2).name)); %H1
i=54;   % H0
[audio, fs0] =audioread(strcat(H0_index((i-(length(H1_index)-2))+2).folder,'\',H0_index((i-(length(H1_index)-2))+2).name));

current_dur  = duration;
start_index  = randi(length(audio)-current_dur*fs0);
audio_cut    = audio(start_index:(start_index+current_dur*fs0-1));
x            = resample(audio_cut, fs, fs0); % Downsampling
N            = length(x); 
        
x_filtered   = filter(BPF,1,x); % Bandpass Filtering

periodogram=zeros(1,N/2);
Laplace_periodogram1=zeros(1,N/2);
Laplace_periodogram2=zeros(1,N/2);
for k = 0:N/2-1
     omega =2*pi*k/N;
     c  = cos(omega*(0:N-1))';
     s  = sin(omega*(0:N-1))';
     H  = [c,s]; % Nx2
     b  = 2* H' * x_filtered/N;   % 2x1
     periodogram(k+1) = N*(norm(b.^2))/4; %LS-LRT - Ordinary Periodogram
     %Laplace I
     theta = ladreg(x_filtered, H, false, [], 1); 
     Laplace_periodogram1(k+1) = N*(norm(theta.^2))/4; %LS-Median - Laplace Periodogram I
     Laplace_periodogram2(k+1) = 1/2*(norm(x_filtered,1) - norm(x_filtered-H*theta,1)); %LAD-Median - Laplace Periodogram II
end
freqs=0:fs/N:fs*(1/2-1/N);

%% Test Periodograms

figure(1)
plot(freqs,periodogram,'r');
grid on
hx=xlabel('$f$ (Hz)'); 
set(hx, 'Interpreter', 'latex');
figure(2)
plot(freqs,Laplace_periodogram1,'g');
grid on
hx=xlabel('$f$ (Hz)'); 
set(hx, 'Interpreter', 'latex');
figure(3)
plot(freqs,Laplace_periodogram2,'b');
grid on
hx=xlabel('$f$ (Hz)'); 
set(hx, 'Interpreter', 'latex');
fc1 = find(periodogram==max(periodogram))*(fs/N);
fc2 = find(Laplace_periodogram1==max(Laplace_periodogram1))*(fs/N);
fc3 = find(Laplace_periodogram2==max(Laplace_periodogram2))*(fs/N);