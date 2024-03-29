%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program calculates the ROC curves for ENF detection using
%
% 1. LAD-LRT - equally spaced thresholds between min and max values of statistic
% 2. LS-LRT - equally spaced thresholds between min and max values of statistic
% 3. naive-LRT - equally spaced thresholds between min and max values of statistic
%
% versus recording length using real-world audio recordings.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear;close all;

%%% Bandpass Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = [0 0.4 0.499 0.4995 0.5 0.5005 0.501 0.6 0.8 1];
M = [0 0 0 0.2 1 0.2 0 0 0 0];
BPF= fir2(1023,F,M);
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
N_thre = 1000;
duration = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
path         = '';
H0_index     = dir(strcat(path,'H0'));
H1_index     = dir(strcat(path,'H1'));
ground_truth = [ones(1,length(H1_index)-2),zeros(1,length(H0_index)-2)];

Test_Statistic1 = zeros(1,length(H1_index)-2+length(H0_index)-2);
Test_Statistic2 = zeros(1,length(H1_index)-2+length(H0_index)-2);
Test_Statistic3 = zeros(1,length(H1_index)-2+length(H0_index)-2);

result1      = zeros((N_thre),(length(H1_index)-2+length(H0_index)-2));  % LAD-LRT - Moving
result2      = zeros((N_thre),(length(H1_index)-2+length(H0_index)-2));  % LS-LRT - Moving 
result3      = zeros((N_thre),(length(H1_index)-2+length(H0_index)-2));  % naive-LRT - Moving

ACC1         = zeros(1,(N_thre));
ACC2         = zeros(1,(N_thre));
ACC3         = zeros(1,(N_thre));

O_TP1 = zeros(1,(N_thre));
O_TN1 = zeros(1,(N_thre));
O_FP1 = zeros(1,(N_thre));
O_FN1 = zeros(1,(N_thre));
O_TP2 = zeros(1,(N_thre));
O_TN2 = zeros(1,(N_thre));
O_FP2 = zeros(1,(N_thre));
O_FN2 = zeros(1,(N_thre));
O_TP3 = zeros(1,(N_thre));
O_TN3 = zeros(1,(N_thre));
O_FP3 = zeros(1,(N_thre));
O_FN3 = zeros(1,(N_thre));


for i = 1:(length(H1_index)-2+length(H0_index)-2)
    disp(['i=',num2str(i)]); 
    if i<=(length(H1_index)-2)
        [audio, fs0] = audioread(strcat(H1_index(i+2).folder,'\',H1_index(i+2).name));
        audio        = audio(:,1)';
    else
        [audio, fs0] = audioread(strcat(H0_index((i-(length(H1_index)-2))+2).folder,'\',H0_index((i-(length(H1_index)-2))+2).name));
        audio        = audio(:,1)';
    end
    
    current_dur  = duration;
    start_index  = randi(length(audio)-current_dur*fs0);
    audio_cut    = audio(start_index:(start_index+current_dur*fs0-1));
    x            = resample(audio_cut, fs, fs0); % Downsampling
    N            = length(x);
    
    x_filtered   = filter(BPF,1,x); % Bandpass Filtering
    
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
    
    Hc1                = [cos(2*pi*T*f_star*(0:N-1))',sin(2*pi*T*f_star*(0:N-1))'];
    Test_Statistic1(i) = (x_filtered*Hc1*theta_star)/((norm(x_filtered).^2)); % LAD-LRT - Moving
          
    Hc2                = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
    Test_Statistic2(i) = 2/N*(x_filtered*Hc2)*(Hc2'*x_filtered')/((norm(x_filtered).^2)); % LS-LRT - Moving
    
    Hc3                 =[cos(2*pi*T*100*(0:N-1))',sin(2*pi*T*100*(0:N-1))'];
    Test_Statistic3(i)  = 2/N*(x_filtered*Hc3)*(Hc3'*x_filtered')/((norm(x_filtered).^2)); % naive-LRT - Moving
end

min_stat1 = min(Test_Statistic1);
max_stat1 = max(Test_Statistic1);
thre1m    = linspace(min_stat1,max_stat1,N_thre); 

min_stat2 = min(Test_Statistic2);
max_stat2 = max(Test_Statistic2);
thre2m    = linspace(min_stat2,max_stat2,N_thre);

min_stat3 = min(Test_Statistic3);
max_stat3 = max(Test_Statistic3);
thre3m    = linspace(min_stat3,max_stat3,N_thre);

for i = 1:(length(H1_index)-2+length(H0_index)-2)
    disp(['i=',num2str(i)]);     
    for j = 1:N_thre
        if Test_Statistic1(i) >= thre1m(j)
            result1(j,i) = 1;
        end
        if Test_Statistic2(i) >= thre2m(j)
            result2(j,i) = 1;
        end
         if Test_Statistic3(i) >= thre3m(j)
            result3(j,i) = 1;
        end
    end
end

for j = 1:N_thre
    [O_TP1(j),O_TN1(j),O_FP1(j),O_FN1(j)] = fun_TP_TN_FP_FN(result1(j,:),ground_truth);
    ACC1(j)              = (O_TP1(j)+O_TN1(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP2(j),O_TN2(j),O_FP2(j),O_FN2(j)] = fun_TP_TN_FP_FN(result2(j,:),ground_truth);
    ACC2(j)              = (O_TP2(j)+O_TN2(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP3(j),O_TN3(j),O_FP3(j),O_FN3(j)] = fun_TP_TN_FP_FN(result3(j,:),ground_truth);
    ACC3(j)              = (O_TP3(j)+O_TN3(j))/(length(H1_index)-2+length(H0_index)-2);
end

figure(1);
pf=plot(O_FP1/(length(H0_index)-2),O_TP1/(length(H1_index)-2),'bo-',...
    O_FP2/(length(H0_index)-2),O_TP2/(length(H1_index)-2),'go:',...
    O_FP3/(length(H0_index)-2),O_TP3/(length(H1_index)-2), 'k--square',...
    [0,1],[0,1]);
pf(1).LineWidth=2;
pf(2).LineWidth=2;
pf(3).LineWidth=2;
grid on;
axis([0 1 0 1]); %equally spaced
hl = legend('LAD-LRT','LS-LRT', 'naive-LRT');
hx = xlabel('$P_{\rm{FA}}$');
hy = ylabel('$P_{\rm{D}}$');
set(hx, 'Interpreter', 'latex');
set(hy, 'Interpreter', 'latex');
set(hl, 'Interpreter', 'latex');

%% Calculate AUC

score1 = sum(result1)/N_thre; 
score2 = sum(result2)/N_thre; 
score3 = sum(result3)/N_thre;

figure(3)
[X1,Y1,T1,AUC1] = perfcurve(ground_truth',score1',1);
[X2,Y2,T2,AUC2] = perfcurve(ground_truth',score2',1);
[X3,Y3,T3,AUC3] = perfcurve(ground_truth',score3',1);

ppf=plot(X1,Y1,'b-',...
    X2,Y2,'g:',...
    X3,Y3,'k--',...
    [0,1],[0,1]); 
ppf(1).LineWidth=2;
ppf(2).LineWidth=2;
ppf(3).LineWidth=2;
grid on     
hl = legend(['LAD-LRT AUC=' num2str(AUC1)],...
    ['LS-LRT AUC=' num2str(AUC2)],...
    ['naive-LRT AUC=' num2str(AUC3)], ...
    'location', 'southeast');
hx = xlabel('$P_{\rm{FA}}$');
hy = ylabel('$P_{\rm{D}}$');
set(hx, 'Interpreter', 'latex');
set(hy, 'Interpreter', 'latex');
set(hl, 'Interpreter', 'latex');
