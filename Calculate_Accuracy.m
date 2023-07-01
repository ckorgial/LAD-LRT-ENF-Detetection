%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program calculates ENF detection accuracies of 
%
% 1. LAD-LRT - Median Threshold
% 2. LS-LRT - Median Threshold
% 3. LS-LRT
% 4. naive-LRT
%
% versus recording length using real-world audio recordings.
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

%%%%% Setting for Figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
duration = 5:10; 
load Threshold_info_5_1_10
thre2 = mean20+2*sqrt(var20);
thre3 = mean30+2*sqrt(var30);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
path         = 'C:\Users\30694\Documents\PHD\Codes\ENF_Detection\Datasets\Recordings\';
H0_index     = dir(strcat(path,'H0'));
H1_index     = dir(strcat(path,'H1'));
ground_truth = [ones(1,length(H1_index)-2),zeros(1,length(H0_index)-2)];

result1      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LAD-LRT - Median
result2      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LS-LRT - Median
result3      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LS-LRT
result4      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % naive-LRT

ACC1         = zeros(1,length(duration));
ACC2         = zeros(1,length(duration));
ACC3         = zeros(1,length(duration));
ACC4         = zeros(1,length(duration));

O_TP1 = zeros(1,length(duration));
O_TN1 = zeros(1,length(duration));
O_FP1 = zeros(1,length(duration));
O_FN1 = zeros(1,length(duration));
O_TP2 = zeros(1,length(duration));
O_TN2 = zeros(1,length(duration));
O_FP2 = zeros(1,length(duration));
O_FN2 = zeros(1,length(duration));
O_TP3 = zeros(1,length(duration));
O_TN3 = zeros(1,length(duration));
O_FP3 = zeros(1,length(duration));
O_FN3 = zeros(1,length(duration));
O_TP4 = zeros(1,length(duration));
O_TN4 = zeros(1,length(duration));
O_FP4 = zeros(1,length(duration));
O_FN4 = zeros(1,length(duration));


for i = 1:(length(H1_index)-2+length(H0_index)-2)
    disp(['i=',num2str(i)]); 
    if i<=(length(H1_index)-2)
        [audio, fs0] = audioread(strcat(H1_index(i+2).folder,'\',H1_index(i+2).name));
        audio        = audio(:,1)';
    else
        [audio, fs0] = audioread(strcat(H0_index((i-(length(H1_index)-2))+2).folder,'\',H0_index((i-(length(H1_index)-2))+2).name));
        audio        = audio(:,1)';
    end
    
    for j = 1:length(duration)
        current_dur  = duration(j);
        start_index  = randi(length(audio)-current_dur*fs0);
        audio_cut    = audio(start_index:(start_index+current_dur*fs0-1));
        x            = resample(audio_cut, fs, fs0); % Downsampling
        N            = length(x); 
        
        x_filtered   = filter(BPF,1,x); % Bandpass Filtering %1 X N
        
        NFFT_full    = max(2^18,2^(nextpow2(N)+2));
        X_filtered   = abs(fft(x_filtered,NFFT_full));
        X_filtered   = X_filtered(1:(end/2+1));
        
        % Initial fc
        fc = find(X_filtered==max(X_filtered))*(fs/NFFT_full);

        % Non-Linear LAD
        convergence_threshold = 1e-4; % Set it properly
        iteration = 0;
        CONT=1;
        while CONT 
            iteration = iteration + 1;
            if iteration == 1
                fcc = fc;
            end
            % Optimization wrt theta
            Hm    = [cos(2*pi*T*fcc*(0:N-1))',sin(2*pi*T*fcc*(0:N-1))'];
            theta = ladreg(x_filtered', Hm, false, [], 1); 
            % Optimization wrt fm
            zmin=1e7;
            for m = 1:99
                fm  = fcc + (m-50)*fs/(60*N);  
                Hm  = [cos(2*pi*T*fm*(0:N-1))',sin(2*pi*T*fm*(0:N-1))'];
                laplacian_error = (x_filtered' - Hm*theta);
                zm  = norm(x_filtered' - Hm*theta,1); % Fix theta
                if zm < zmin
                    fcc_new=fm;
                    zmin=zm; 
                end   
            end
            relative_difference = abs(fcc - fcc_new)/fcc ;
            if relative_difference > convergence_threshold
                fcc=fcc_new;
            else
                f_star=fcc;
                CONT=0;
            end
        end
        
        theta_star = ladreg(x_filtered', [cos(2*pi*T*f_star*(0:N-1))',sin(2*pi*T*f_star*(0:N-1))'],false,[],1);
        
        Hc1             = [cos(2*pi*T*f_star*(0:N-1))',sin(2*pi*T*f_star*(0:N-1))']; 
        Test_Statistic1 = (x_filtered*Hc1*theta_star)/((norm(x_filtered).^2)); % LAD-LRT - Median
        p1(j,i)         = Test_Statistic1;
        
        Hc2              = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
        Test_Statistic2  = 2/N*(x_filtered*Hc2)*(Hc2'*x_filtered')/((norm(x_filtered).^2)); % LS-LRT - Median
        p2(j,i)          = Test_Statistic2;
        
        Hc3              = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
        Test_Statistic3  = 2/N*(x_filtered*Hc3)*(Hc3'*x_filtered')/((norm(x_filtered).^2)); % LS-LRT

        Hc4              =[cos(2*pi*T*100*(0:N-1))',sin(2*pi*T*100*(0:N-1))'];
        Test_Statistic4  = 2/N*(x_filtered*Hc4)*(Hc4'*x_filtered')/((norm(x_filtered).^2)); % naive-LRT
        
        if Test_Statistic3 >= thre2(j)
            result3(j,i) = 1; 
        end
         if Test_Statistic4 >= thre3(j)
            result4(j,i) = 1;
        end 
    end
end

thre1m = median(p1,2); 
thre2m = median(p2,2); 

for j = 1:length(duration)
  for i = 1:(length(H1_index)-2+length(H0_index)-2)    
     if (p1(j,i) >= thre1m(j))
        result1(j,i) = 1;
     end
     if (p2(j,i) >= thre2m(j))
        result2(j,i) = 1;
     end   
  end
end

for j = 1:length(duration)
    [O_TP1(j),O_TN1(j),O_FP1(j),O_FN1(j)] = fun_TP_TN_FP_FN(result1(j,:),ground_truth);
    ACC1(j)              = (O_TP1(j)+O_TN1(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP2(j),O_TN2(j),O_FP2(j),O_FN2(j)] = fun_TP_TN_FP_FN(result2(j,:),ground_truth);
    ACC2(j)              = (O_TP2(j)+O_TN2(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP3(j),O_TN3(j),O_FP3(j),O_FN3(j)] = fun_TP_TN_FP_FN(result3(j,:),ground_truth);
    ACC3(j)              = (O_TP3(j)+O_TN3(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP4(j),O_TN4(j),O_FP4(j),O_FN4(j)] = fun_TP_TN_FP_FN(result4(j,:),ground_truth);
    ACC4(j)              = (O_TP4(j)+O_TN4(j))/(length(H1_index)-2+length(H0_index)-2);
end

%% Calculate Accuracy

figure(1);
pf=plot(duration,ACC1*100,'bo-',duration,ACC2*100,'gx:',duration,ACC3*100,'r+-.', duration, ACC4*100,'k--square' );
pf(1).LineWidth=2;
pf(2).LineWidth=2;
pf(3).LineWidth=2;
pf(4).LineWidth=2;

grid on;
hl = legend('LAD-LRT-Median','LS-LRT-Median','LS-LRT', 'naive-LRT');
hx = xlabel('$N/f_{\rm{S}}$');
hy = ylabel('Accuracy ($\%$)');
set(hx, 'Interpreter', 'latex');
set(hy, 'Interpreter', 'latex');
set(hl, 'Interpreter', 'latex');