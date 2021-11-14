% Target Tracking: Computer Exercise 5

%% 1 Consistency

%% Initialization
% True Target Model 

T = 1; %Sample Interval

% State Transition Model 

F = [eye(2)  T*eye(2); %State Transition Matrix
     zeros(2) eye(2)];

G = [T^2/2*eye(2); %Process Noise Gain Matrix
     T*eye(2)];

sigma_a = 2; %Acceleration Noise Standard Deviation
Q = (sigma_a)^2*eye(2); %Process Noise Covariance
 
% Measurement Model 

H = [eye(2) zeros(2)]; %Measurement Matrix

sigma_v = 20; %Measurement Noise Standard Deviation
R = (sigma_v)^2*eye(2); %Measurement Noise Covariance

% Filter Parameters

Filter_Parameters.T = T;
Filter_Parameters.H=  H;
Filter_Parameters.Q = Q;
Filter_Parameters.R = R;

% Simulation Parameters

Step_Num = 100;
MC_Num = 100; %Monte Carlo Number
eps = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Estimate Initialization

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);

StateEstimates = zeros(4,Step_Num);
StateEstimates(:,1) = x0_bar;   
StateEstimatesCov = zeros(4,4,Step_Num);
StateEstimatesCov(:,:,1) = P0_bar;

%% Monte Carlo Simulation

for MC = 1:MC_Num
        
    x_k = zeros(4,Step_Num);
    x_k(:,1) = x0_bar + chol(P0_bar)*randn(4,1);
    
    y_k = zeros(2,Step_Num);
    y_k(:,1) = H*x_k(:,1)+ chol(R)*randn(2,1);
    
    err = x_k(:,1)-StateEstimates(:,1);
    eps(MC,1) = err'*inv(StateEstimatesCov(:,:,1))*err;
    
    
    for i=1:99
        
        % Prediction:
        
        [StatePrediction, ...
            StatePredictionCov, ...
            OutputPrediction, ...
            OutputPredictionCov, ...
            KalmanGain] = kf_pre(StateEstimates(:,i),...
            StateEstimatesCov(:,:,i),...
            Filter_Parameters);
        
        % True Target Data Generation
        x_k(:,i+1) = F*x_k(:,i) + G*(sigma_a*randn(2,1));
        % Measurement Generation
        y_k(:,i+1) = H*x_k(:,i+1) + chol(R)*randn(2,1);
        
        % Estimation:
        [...
            StateEstimates(:,i+1), ...
            StateEstimatesCov(:,:,i+1)] = kf_est(...
            StatePrediction,...
            StatePredictionCov,...
            OutputPrediction, ...
            OutputPredictionCov, ...
            KalmanGain,...
            y_k(:,i+1));
        
        err = x_k(:,i+1)-StateEstimates(:,i+1);
        eps(MC,i+1) = err'*inv(StateEstimatesCov(:,:,i+1))*err;
        
    end
end

% eps_mean = mean(eps,1);
% thresh_min = chi2inv(0.005,MC_Num*4)/MC_Num;
% thresh_max = chi2inv(0.995,MC_Num*4)/MC_Num;
% plot(eps_mean)
% hold on
% plot(repmat(thresh_min,1,100))
% plot(repmat(thresh_max,1,100))

%% 2 Track Fusion

%% Initialization
% True Target Model

T = 1; %Sample Interval

% State Transition Model

F = [eye(2)  T*eye(2); %State Transition Matrix
     zeros(2) eye(2)];

G = [T^2/2*eye(2); %Process Noise Gain Matrix
     T*eye(2)];

sigma_a = 2; %Acceleration Noise Standard Deviation
Q = (sigma_a)^2*eye(2); %Process Noise Covariance

% Measurement Model 

H = [eye(2) zeros(2)]; %Measurement Matrix

sigma_v = 20; %Measurement Noise Standard Deviation
R = (sigma_v)^2*eye(2); %Measurement Noise Covariance
% Simulation Parameters

Step_Num = 100;
MC_Num = 100; %Monte Carlo Number

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);


x_k = zeros(4,Step_Num,MC_Num);
y_k_1 = zeros(2,Step_Num,MC_Num);
y_k_2 = zeros(2,Step_Num,MC_Num);

chol_R = chol(R);
chol_P0_bar = chol(P0_bar);

for MC = 1:MC_Num
 x_k(:,1,MC) = x0_bar + chol_P0_bar*randn(4,1); % True Target Data Generation
 y_k_1(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
 y_k_2(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
    for i=2:Step_Num
        x_k(:,i,MC) = F*x_k(:,i-1,MC) + G*(sigma_a*randn(2,1)); % True Target Data Generation
        y_k_1(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
        y_k_2(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
    end
end

yk_centralized = [y_k_1;
                  y_k_2];
              
%% Centralized Solution 

eps_centralized = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Filter Parameters
Filter_Parameters.T = T;
Filter_Parameters.H= [H;
                      H];
Filter_Parameters.Q = Q;
Filter_Parameters.R = blkdiag(R,R);

% Estimate Initialization
StateEstimates = zeros(4,Step_Num);
StateEstimates(:,1) = x0_bar;   
StateEstimatesCov = zeros(4,4,Step_Num);
StateEstimatesCov(:,:,1) = P0_bar;
err_cent = zeros(4,Step_Num,MC_Num);
% Monte Carlo Simulation

for MC = 1:MC_Num
    
    err_cent(:,1,MC) = x_k(:,1,MC)-StateEstimates(:,1);
    eps_centralized(MC,1) = err_cent(:,1,MC)'*inv(StateEstimatesCov(:,:,1))*err_cent(:,1,MC);
    
    for i=1:Step_Num-1
        
        % Prediction:
        
        [StatePrediction, ...
            StatePredictionCov, ...
            OutputPrediction, ...
            OutputPredictionCov, ...
            KalmanGain] = kf_pre(StateEstimates(:,i),...
            StateEstimatesCov(:,:,i),...
            Filter_Parameters);

        % Estimation:
        [...
            StateEstimates(:,i+1), ...
            StateEstimatesCov(:,:,i+1)] = kf_est(...
            StatePrediction,...
            StatePredictionCov,...
            OutputPrediction, ...
            OutputPredictionCov, ...
            KalmanGain,...
            yk_centralized(:,i+1,MC));

        err_cent(:,i+1,MC) = x_k(:,i+1,MC)-StateEstimates(:,i+1);
        eps_centralized(MC,i+1) = err_cent(:,i+1,MC)'*inv(StateEstimatesCov(:,:,i+1))*err_cent(:,i+1,MC);

    end
end

% Filter Parameters

Filter_Parameters.T = T;
Filter_Parameters.H = H;
Filter_Parameters.Q = Q;
Filter_Parameters.R = R;

%% Decentralized Solution

eps_decentralized = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Estimate Initialization

StateEstimatesDecentralized = zeros(4,Step_Num,2);
StateEstimatesDecentralized(:,1,1) = x0_bar;
StateEstimatesDecentralized(:,1,2) = x0_bar;
StateEstimatesCovDecentralized = zeros(4,4,Step_Num,2);
StateEstimatesCovDecentralized(:,:,1,1) = P0_bar;
StateEstimatesCovDecentralized(:,:,1,2) = P0_bar;
err_decent = zeros(4,Step_Num,MC_Num);

for MC = 1:MC_Num
   
    for i=1:Step_Num-1
        if  mod(i,2) == 1
            InvLocalAgent1_Cov = inv(StateEstimatesCovDecentralized(:,:,i,1));
            InvLocalAgent2_Cov = inv(StateEstimatesCovDecentralized(:,:,i,2));
            StateEstimatesCovDecentralized(:,:,i,2) = inv(InvLocalAgent1_Cov+InvLocalAgent2_Cov);
            StateEstimatesDecentralized(:,i,2) = StateEstimatesCovDecentralized(:,:,i,2)*(InvLocalAgent1_Cov*StateEstimatesDecentralized(:,i,1)+...
                                                                                          InvLocalAgent2_Cov*StateEstimatesDecentralized(:,i,2));
        end
              
        err_decent(:,i,MC) = x_k(:,i,MC)-StateEstimatesDecentralized(:,i,2);
        eps_decentralized(MC,i) = err_decent(:,i,MC)'*inv(StateEstimatesCovDecentralized(:,:,i,2))*err_decent(:,i,MC);
        
        for l=1:2
        % Prediction:
        
        [StatePrediction, ...
         StatePredictionCov, ...
         OutputPrediction, ...
         OutputPredictionCov, ...
         KalmanGain] = kf_pre(StateEstimatesDecentralized(:,i,l),...
                              StateEstimatesCovDecentralized(:,:,i,l),...
                              Filter_Parameters);

        % Estimation:
        [...
            StateEstimatesDecentralized(:,i+1,l), ...
            StateEstimatesCovDecentralized(:,:,i+1,l)] = kf_est(...
                                                StatePrediction,...
                                                StatePredictionCov,...
                                                OutputPrediction, ...
                                                OutputPredictionCov, ...
                                                KalmanGain,...
                                                yk_centralized(2*l-1:2*l,i+1,MC));
        end

    end

       err_decent(:,end,MC) = x_k(:,end,MC)-StateEstimatesDecentralized(:,end,2);
       eps_decentralized(MC,end) = err_decent(:,end,MC)'*inv(StateEstimatesCovDecentralized(:,:,end,2))*err_decent(:,end,MC);
       
end 
%% Channel Filter

eps_channel_filter = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Estimate Initialization

StateEstimatesChannelFilter = zeros(4,Step_Num,2);
StateEstimatesChannelFilter(:,1,1) = x0_bar;
StateEstimatesChannelFilter(:,1,2) = x0_bar;
StateEstimatesCovChannelFilter = zeros(4,4,Step_Num,2);
StateEstimatesCovChannelFilter(:,:,1,1) = P0_bar;
StateEstimatesCovChannelFilter(:,:,1,2) = P0_bar;
err_channel = zeros(4,Step_Num,MC_Num);

for MC = 1:MC_Num
   
    for i=1:Step_Num-1
        if  i~= 1 && mod(i,2) == 1
            ExtrapolatedStateEstimate = StateEstimatesChannelFilter(:,i-2,1);
            ExtrapolatedStateEstimateCov = StateEstimatesCovChannelFilter(:,:,i-2,1);
            for l=1:2
                % Prediction:
                
                [ExtrapolatedStateEstimate, ...
                 ExtrapolatedStateEstimateCov, ...
                 ~, ~, ~] = kf_pre(ExtrapolatedStateEstimate,...
                                   ExtrapolatedStateEstimateCov,...
                                   Filter_Parameters);
            end
            
            InvLocalAgent1_Cov = inv(StateEstimatesCovChannelFilter(:,:,i,1));
            InvLocalAgent2_Cov = inv(StateEstimatesCovChannelFilter(:,:,i,2));
            InvExtrapolatedStateEstimateCov = inv(ExtrapolatedStateEstimateCov);
            StateEstimatesCovChannelFilter(:,:,i,2) = inv(InvLocalAgent1_Cov+InvLocalAgent2_Cov-InvExtrapolatedStateEstimateCov);
            StateEstimatesChannelFilter(:,i,2) = StateEstimatesCovChannelFilter(:,:,i,2)*(InvLocalAgent1_Cov*StateEstimatesChannelFilter(:,i,1)+...
                                                                                          InvLocalAgent2_Cov*StateEstimatesChannelFilter(:,i,2)-...
                                                                                          InvExtrapolatedStateEstimateCov*ExtrapolatedStateEstimate);            
        end
              
        err_channel(:,i,MC) = x_k(:,i,MC)-StateEstimatesChannelFilter(:,i,2);
        eps_channel_filter(MC,i) = err_channel(:,i,MC)'*inv(StateEstimatesCovChannelFilter(:,:,i,2))*err_channel(:,i,MC);
        
        for l=1:2
        % Prediction:
        
        [StatePrediction, ...
         StatePredictionCov, ...
         OutputPrediction, ...
         OutputPredictionCov, ...
         KalmanGain] = kf_pre(StateEstimatesChannelFilter(:,i,l),...
                              StateEstimatesCovChannelFilter(:,:,i,l),...
                              Filter_Parameters);

        % Estimation:
        [...
            StateEstimatesChannelFilter(:,i+1,l), ...
            StateEstimatesCovChannelFilter(:,:,i+1,l)] = kf_est(...
                                                StatePrediction,...
                                                StatePredictionCov,...
                                                OutputPrediction, ...
                                                OutputPredictionCov, ...
                                                KalmanGain,...
                                                yk_centralized(2*l-1:2*l,i+1,MC));
        end

    end

       err_channel(:,end,MC) = x_k(:,end,MC)-StateEstimatesChannelFilter(:,end,2);
       eps_channel_filter(MC,end) = err_channel(:,end,MC)'*inv(StateEstimatesCovChannelFilter(:,:,end,2))*err_channel(:,end,MC);
       
       
end     

%% Covariance Intersection

eps_covariance_intersection = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Estimate Initialization

StateEstimatesCI = zeros(4,Step_Num,2);
StateEstimatesCI(:,1,1) = x0_bar;
StateEstimatesCI(:,1,2) = x0_bar;
StateEstimatesCovCI = zeros(4,4,Step_Num,2);
StateEstimatesCovCI(:,:,1,1) = P0_bar;
StateEstimatesCovCI(:,:,1,2) = P0_bar;
err_cov_int = zeros(4,Step_Num,MC_Num);

for MC = 1:MC_Num
    
    for i=1:Step_Num-1
        if  mod(i,2) == 1
            InvLocalAgent1_Cov = inv(StateEstimatesCovCI(:,:,i,1));
            InvLocalAgent2_Cov = inv(StateEstimatesCovCI(:,:,i,2));
            w = Covariance_Intersection(InvLocalAgent1_Cov,InvLocalAgent2_Cov);
            StateEstimatesCovCI(:,:,i,2) = inv(w*InvLocalAgent1_Cov+(1-w)*InvLocalAgent2_Cov);
            StateEstimatesCI(:,i,2) = StateEstimatesCovCI(:,:,i,2)*(w*InvLocalAgent1_Cov*StateEstimatesCI(:,i,1)+...
                                                                    (1-w)*InvLocalAgent2_Cov*StateEstimatesCI(:,i,2));                                         
        end
        
        err_cov_int(:,i,MC) = x_k(:,i,MC)-StateEstimatesCI(:,i,2);
        eps_covariance_intersection(MC,i) = err_cov_int(:,i,MC)'*inv(StateEstimatesCovCI(:,:,i,2))*err_cov_int(:,i,MC);
        
        for l=1:2
            % Prediction:
            
            [StatePrediction, ...
                StatePredictionCov, ...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain] = kf_pre(StateEstimatesCI(:,i,l),...
                StateEstimatesCovCI(:,:,i,l),...
                Filter_Parameters);
            
            % Estimation:
            [...
                StateEstimatesCI(:,i+1,l), ...
                StateEstimatesCovCI(:,:,i+1,l)] = kf_est(...
                StatePrediction,...
                StatePredictionCov,...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain,...
                yk_centralized(2*l-1:2*l,i+1,MC));
        end
    end
    
    err_cov_int(:,end,MC) = x_k(:,end,MC)-StateEstimatesCI(:,end,2);
    eps_covariance_intersection(MC,end) = err_cov_int(:,end,MC)'*inv(StateEstimatesCovCI(:,:,end,2))*err_cov_int(:,end,MC);
end

time = 0:Step_Num-1*T;
err_cent_mean = sqrt(mean(err_cent.^2,3));
err_decent_mean = sqrt(mean(err_decent.^2,3));
err_channel_mean = sqrt(mean(err_channel.^2,3));
err_cov_int_mean = sqrt(mean(err_cov_int.^2,3));

eps_centralized_mean = mean(eps_centralized,1);
eps_decentralized_mean = mean(eps_decentralized,1);
eps_channel_filter_mean = mean(eps_channel_filter,1);
eps_covariance_intersection_mean = mean(eps_covariance_intersection,1);

figure;
thresh_min = chi2inv(0.005,MC_Num*4)/MC_Num;
thresh_max = chi2inv(0.995,MC_Num*4)/MC_Num;
hold on
plot(time,eps_centralized_mean)
plot(time,eps_decentralized_mean)
plot(time,eps_channel_filter_mean)
plot(time,eps_covariance_intersection_mean)
plot(time,repmat(thresh_min,1,100))
plot(time,repmat(thresh_max,1,100))
legend({'Centralized', 'Decentralized','Channel Filter','Covariance Intersection','Threshold Min','Threshold Max'}, 'fontsize', 10);
ylabel('NEES', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

figure;
hold on;
plot(time,err_cent_mean(1,:));
plot(time,err_decent_mean(1,:));
plot(time,err_channel_mean(1,:));
plot(time,err_cov_int_mean(1,:));
legend({'Centralized', 'Decentralized','Channel Filter','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Position X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

figure;
hold on;
plot(time,err_cent_mean(2,:));
plot(time,err_decent_mean(2,:));
plot(time,err_channel_mean(2,:));
plot(time,err_cov_int_mean(2,:));
legend({'Centralized', 'Decentralized','Channel Filter','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Position Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time,err_cent_mean(3,:));
plot(time,err_decent_mean(3,:));
plot(time,err_channel_mean(3,:));
plot(time,err_cov_int_mean(3,:));
legend({'Centralized', 'Decentralized','Channel Filter','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Velocity X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time,err_cent_mean(4,:));
plot(time,err_decent_mean(4,:));
plot(time,err_channel_mean(4,:));
plot(time,err_cov_int_mean(4,:));
legend({'Centralized', 'Decentralized','Channel Filter','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Velocity Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
