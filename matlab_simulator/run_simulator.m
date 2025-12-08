% Simple script to run EMG Robot Arm Simulator
% 
% Usage:
%   1. Start FastAPI server in terminal:
%      ./start_server.sh
%   
%   2. Run this script in MATLAB:
%      run_simulator
%   
%   3. Watch the robot arm move!
%
% Press Ctrl+C to stop

clear;
clc;

fprintf('========================================\n');
fprintf('EMG Robot Arm Simulator\n');
fprintf('========================================\n\n');

% Configuration
SERVER_URL = 'http://localhost:8000';
DURATION = 60;  % seconds (set to inf for infinite)

% Create simulator
fprintf('Initializing simulator...\n');
sim = RobotArmSimulator(SERVER_URL);

fprintf('Starting visualization...\n');
fprintf('Duration: %d seconds\n', DURATION);
fprintf('Press Ctrl+C to stop early\n\n');

% Start simulation
sim.start(DURATION);

fprintf('\nSimulation complete!\n');