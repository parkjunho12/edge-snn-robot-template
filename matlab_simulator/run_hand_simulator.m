% Run Robot Hand Simulator
% Complete hand visualization with all 5 fingers
%
% Usage:
%   1. Start FastAPI server: ./start_server.sh
%   2. Run this script in MATLAB
%   3. Watch the robot hand move!

clear;
clc;

fprintf('========================================\n');
fprintf('EMG Robot Hand Simulator\n');
fprintf('Full 5-Finger Control\n');
fprintf('========================================\n\n');

% Configuration
SERVER_URL = 'http://localhost:8000';
DURATION = 60;  % seconds (set to inf for infinite)

% Create simulator
fprintf('Initializing hand simulator...\n');
fprintf('- 5 fingers with 3 joints each\n');
fprintf('- Wrist rotation and flexion\n');
fprintf('- Real-time EMG control\n\n');

sim = RobotHandSimulator(SERVER_URL);

fprintf('Starting visualization...\n');
fprintf('Duration: %d seconds\n', DURATION);
fprintf('Press Ctrl+C to stop early\n\n');

% Start simulation

pause(1); sim.updateHand(0, 0.95, 1.0);  % Rest
pause(1); sim.updateHand(1, 0.95, 1.0);  % Hand Open
pause(1); sim.updateHand(2, 0.95, 1.0);  % Fist
pause(1); sim.updateHand(3, 0.95, 1.0);  % Wrist Flex
pause(1); sim.updateHand(4, 0.95, 1.0);  % Wrist Extend
pause(1); sim.updateHand(5, 0.95, 1.0);  % Pinch
pause(1); sim.updateHand(6, 0.95, 1.0);  % Point

sim.start(DURATION);


fprintf('\n✓ Simulation complete!\n');