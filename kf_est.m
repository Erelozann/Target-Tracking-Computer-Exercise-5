
function [...
    StateEstimate, ...
    StateEstimateCov] = kf_est(StatePrediction,...
                               StatePredictionCov,...
                               OutputPrediction, ...
                               OutputPredictionCov, ...
                               KalmanGain,...
                               Measurement)
                           
    Innovation = Measurement-OutputPrediction;
    K = KalmanGain;
    S = OutputPredictionCov;

    % State Update
    x = StatePrediction + K*Innovation;
    
    % Covariance Update
    P = StatePredictionCov - K*S*K';

    % Simetrikligi Koru:
    P = (P + P')/2;

    % Output
    StateEstimate = x;
    StateEstimateCov = P;

end