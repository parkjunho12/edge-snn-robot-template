% EMG Robot Arm Simulator
% Visualizes robot arm movements based on EMG inference predictions
% Connects to FastAPI server via HTTP and receives streaming predictions
%
% Usage:
%   1. Start FastAPI server: ./start_server.sh
%   2. Run this script in MATLAB
%   3. Watch robot arm move based on EMG predictions
%
% Author: EMG Inference System v0.3
% Date: 2025

classdef RobotArmSimulator < handle
    % Robot Arm Simulator for EMG Control Visualization
    % This class connects to the inference server and visualizes
    % robot arm movements in real-time
    
    properties
        % Server connection
        serverURL = 'http://localhost:8000';
        
        % Robot configuration (5-DOF robot arm)
        numJoints = 5;
        jointAngles = [0, 0, 0, 0, 0];  % Current joint angles (radians)
        targetAngles = [0, 0, 0, 0, 0]; % Target joint angles
        linkLengths = [0.1, 0.3, 0.25, 0.15, 0.1]; % Link lengths (meters)
        
        % Visualization
        fig;           % Figure handle
        ax;            % Axis handle
        robotPlot;     % Robot plot handles
        textHandles;   % Text display handles
        
        % Prediction history
        predictionHistory = [];
        confidenceHistory = [];
        latencyHistory = [];
        maxHistoryLength = 100;
        
        % Animation parameters
        animationSpeed = 0.1; % Interpolation speed (0-1)
        updateRate = 30;      % Hz
        
        % Gesture mapping (same as ROS2 bridge)
        gestureMappings = containers.Map('KeyType', 'double', 'ValueType', 'any');
        
        % Status
        isRunning = false;
        frameCount = 0;
        startTime;
    end
    
    methods
        function obj = RobotArmSimulator(serverURL)
            % Constructor
            % Args:
            %   serverURL: URL of FastAPI server (optional)
            
            if nargin > 0
                obj.serverURL = serverURL;
            end
            
            % Initialize visualization
            obj.initializeVisualization();
            
            % Initialize gesture mappings
            obj.initializeGestureMappings();
            
            % Test server connection
            obj.testConnection();
        end
        
        function initializeGestureMappings(obj)
            % Initialize gesture mappings (same as ROS2 bridge)
            
            % Gesture 0: Rest
            obj.gestureMappings(0) = struct(...
                'name', 'Rest', ...
                'angles', [0.0, 0.0, 0.0, 0.0, 0.0], ...
                'color', [0.5, 0.5, 0.5]);
            
            % Gesture 1: Hand Open
            obj.gestureMappings(1) = struct(...
                'name', 'Hand Open', ...
                'angles', [0.0, 0.5, 0.5, 0.5, 0.5], ...
                'color', [0.0, 1.0, 0.0]);
            
            % Gesture 2: Hand Close
            obj.gestureMappings(2) = struct(...
                'name', 'Hand Close', ...
                'angles', [0.0, -0.5, -0.5, -0.5, -0.5], ...
                'color', [1.0, 0.0, 0.0]);
            
            % Gesture 3: Wrist Flex
            obj.gestureMappings(3) = struct(...
                'name', 'Wrist Flex', ...
                'angles', [0.5, 0.0, 0.0, 0.0, 0.0], ...
                'color', [0.0, 0.0, 1.0]);
            
            % Gesture 4: Wrist Extend
            obj.gestureMappings(4) = struct(...
                'name', 'Wrist Extend', ...
                'angles', [-0.5, 0.0, 0.0, 0.0, 0.0], ...
                'color', [1.0, 1.0, 0.0]);
            
            % Gesture 5: Pinch
            obj.gestureMappings(5) = struct(...
                'name', 'Pinch', ...
                'angles', [0.0, -0.3, -0.3, 0.5, 0.5], ...
                'color', [1.0, 0.0, 1.0]);
            
            % Gesture 6: Point
            obj.gestureMappings(6) = struct(...
                'name', 'Point', ...
                'angles', [0.0, 0.5, -0.5, -0.5, -0.5], ...
                'color', [0.0, 1.0, 1.0]);
        end
        
        function initializeVisualization(obj)
            % Initialize 3D visualization of robot arm
            
            obj.fig = figure('Name', 'EMG Robot Arm Simulator', ...
                           'NumberTitle', 'off', ...
                           'Position', [100, 100, 1200, 800], ...
                           'Color', [0.95, 0.95, 0.95]);
            
            % Create main 3D plot
            obj.ax = subplot(2, 2, [1, 3]);
            hold(obj.ax, 'on');
            grid(obj.ax, 'on');
            axis(obj.ax, 'equal');
            xlabel(obj.ax, 'X (m)');
            ylabel(obj.ax, 'Y (m)');
            zlabel(obj.ax, 'Z (m)');
            title(obj.ax, 'Robot Arm - EMG Control');
            view(obj.ax, 45, 30);
            
            % Set axis limits
            axisLimit = sum(obj.linkLengths);
            xlim(obj.ax, [-axisLimit, axisLimit]);
            ylim(obj.ax, [-axisLimit, axisLimit]);
            zlim(obj.ax, [0, axisLimit]);
            
            % Initialize robot plot
            obj.robotPlot = plot3(obj.ax, 0, 0, 0, 'o-', ...
                                 'LineWidth', 3, ...
                                 'MarkerSize', 10, ...
                                 'MarkerFaceColor', 'b');
            
            % Create prediction history plot
            subplot(2, 2, 2);
            title('Prediction History');
            xlabel('Frame');
            ylabel('Gesture Class');
            ylim([0, 7]);
            grid on;
            
            % Create latency plot
            subplot(2, 2, 4);
            title('Inference Latency');
            xlabel('Frame');
            ylabel('Latency (ms)');
            grid on;
            
            % Add text displays
            obj.textHandles = struct();
            obj.textHandles.status = annotation('textbox', [0.02, 0.95, 0.3, 0.04], ...
                'String', 'Status: Initializing...', ...
                'EdgeColor', 'none', ...
                'FontSize', 12, ...
                'FontWeight', 'bold');
            
            obj.textHandles.gesture = annotation('textbox', [0.02, 0.90, 0.3, 0.04], ...
                'String', 'Gesture: None', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.confidence = annotation('textbox', [0.02, 0.85, 0.3, 0.04], ...
                'String', 'Confidence: 0%', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.fps = annotation('textbox', [0.02, 0.80, 0.3, 0.04], ...
                'String', 'FPS: 0', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.latency = annotation('textbox', [0.02, 0.75, 0.3, 0.04], ...
                'String', 'Latency: 0 ms', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            drawnow;
        end
        
        function success = testConnection(obj)
            % Test connection to FastAPI server
            
            try
                url = sprintf('%s/health', obj.serverURL);
                response = webread(url);
                
                if strcmp(response.status, 'ok')
                    fprintf('✓ Connected to server at %s\n', obj.serverURL);
                    obj.updateStatus('Connected', [0, 1, 0]);
                    success = true;
                else
                    warning('Server returned unexpected status');
                    success = false;
                end
            catch ME
                warning('Failed to connect to server: %s', ME.message);
                obj.updateStatus('Disconnected', [1, 0, 0]);
                success = false;
            end
        end
        
        function start(obj, duration)
            % Start streaming inference and visualization
            % Args:
            %   duration: Streaming duration in seconds (default: inf)
            
            if nargin < 2
                duration = inf;
            end
            
            obj.isRunning = true;
            obj.frameCount = 0;
            obj.startTime = tic;
            
            fprintf('Starting robot arm simulator...\n');
            fprintf('Press Ctrl+C to stop\n\n');
            
            obj.updateStatus('Running', [0, 1, 0]);
            
            % Start streaming loop
            obj.streamingLoop(duration);
        end
        
        function streamingLoop(obj, duration)
            % Main streaming loop - connects to /infer/stream endpoint
            
            try
                % Prepare streaming request
                url = sprintf('%s/infer/stream', obj.serverURL);
                options = weboptions('ContentType', 'json', ...
                                    'Timeout', duration + 10);
                
                % Request configuration
                config = struct('duration_seconds', duration, ...
                              'use_tensorrt', false, ...
                              'fps', 30, ...
                              'preprocess', true);
                
                % Note: MATLAB doesn't support Server-Sent Events natively
                % So we'll use a polling approach instead
                fprintf('Starting polling mode (MATLAB limitation with SSE)\n');
                obj.pollingLoop(duration);
                
            catch ME
                fprintf('Error in streaming: %s\n', ME.message);
                obj.updateStatus('Error', [1, 0, 0]);
            end
        end
        
        function pollingLoop(obj, duration)
            % Polling-based inference loop
            % Calls /infer/tensorrt endpoint repeatedly
            
            frameInterval = 1.0 / obj.updateRate;
            
            while obj.isRunning && toc(obj.startTime) < duration
                frameStart = tic;
                
                try
                    % Call single inference endpoint
                    url = sprintf('%s/infer', obj.serverURL);
                    options = weboptions('ContentType', 'json', 'Timeout', 5);
                    
                    % Prepare request
                    request = struct('encoding_type', 'rate', 'model_prefix', 'tcn', 'device', 'cpu');
                    
                    % Send request
                    response = webwrite(url, request, options);
                    
                    % Parse response
                    prediction = str2double(response.pred_idx);
                    confidence = str2double(response.conf);
                    latency = str2double(response.latency_ms);
                    
                    % Update robot
                    obj.updateRobot(prediction, confidence, latency);
                    
                    obj.frameCount = obj.frameCount + 1;
                    
                catch ME
                    fprintf('Frame error: %s\n', ME.message);
                end
                
                % Maintain frame rate
                frameElapsed = toc(frameStart);
                if frameElapsed < frameInterval
                    pause(frameInterval - frameElapsed);
                end
                
                % Allow MATLAB to process events
                drawnow;
            end
            
            obj.stop();
        end
        
        function updateRobot(obj, prediction, confidence, latency)
            % Update robot visualization based on prediction
            % Args:
            %   prediction: Gesture class (0-6)
            %   confidence: Prediction confidence (0-1)
            %   latency: Inference latency in ms
            
            % Update history
            obj.predictionHistory(end+1) = prediction;
            obj.confidenceHistory(end+1) = confidence;
            obj.latencyHistory(end+1) = latency;
            
            % Limit history length
            if length(obj.predictionHistory) > obj.maxHistoryLength
                obj.predictionHistory(1) = [];
                obj.confidenceHistory(1) = [];
                obj.latencyHistory(1) = [];
            end
            
            % Get gesture mapping
            if obj.gestureMappings.isKey(prediction)
                gesture = obj.gestureMappings(prediction);
                obj.targetAngles = gesture.angles;
                gestureName = gesture.name;
                gestureColor = gesture.color;
            else
                gestureName = 'Unknown';
                gestureColor = [0.5, 0.5, 0.5];
            end
            
            % Smoothly interpolate to target angles
            obj.jointAngles = obj.jointAngles + ...
                obj.animationSpeed * (obj.targetAngles - obj.jointAngles);
            
            % Update 3D visualization
            obj.updateVisualization(gestureColor);
            
            % Update text displays
            obj.textHandles.gesture.String = sprintf('Gesture: %s (%d)', gestureName, prediction);
            obj.textHandles.confidence.String = sprintf('Confidence: %.1f%%', confidence * 100);
            obj.textHandles.latency.String = sprintf('Latency: %.2f ms', latency);
            
            % Calculate FPS
            if obj.frameCount > 0
                elapsedTime = toc(obj.startTime);
                fps = obj.frameCount / elapsedTime;
                obj.textHandles.fps.String = sprintf('FPS: %.1f', fps);
            end
            
            % Update history plots
            obj.updateHistoryPlots();
        end
        
        function updateVisualization(obj, color)
            % Update 3D robot arm visualization
            % Args:
            %   color: RGB color for the robot

            % Forward kinematics - compute joint positions
            positions = zeros(obj.numJoints + 1, 3);
            positions(1, :) = [0, 0, 0]; % Base position

            % Current transformation matrix
            T = eye(4);

            for i = 1:obj.numJoints
                % Joint rotation (around Z axis)
                angle = obj.jointAngles(i);
                length = obj.linkLengths(i);

                % Rotation around Z
                Rz = [cos(angle), -sin(angle), 0, 0;
                    sin(angle),  cos(angle), 0, 0;
                    0,           0,          1, 0;
                    0,           0,          0, 1];

                % Translation along local X axis
                Tx = [1, 0, 0, length;
                    0, 1, 0, 0;
                    0, 0, 1, 0;
                    0, 0, 0, 1];

                % Update transformation: rotate then translate
                T = T * Rz * Tx;

                % Store joint position
                positions(i+1, :) = T(1:3, 4)';
            end

            % Update plot
            set(obj.robotPlot, 'XData', positions(:, 1), ...
                            'YData', positions(:, 2), ...
                            'ZData', positions(:, 3), ...
                            'Color', color, ...
                            'MarkerFaceColor', color);
        end
        
        function updateHistoryPlots(obj)
            % Update prediction and latency history plots
            
            % Prediction history
            subplot(2, 2, 2);
            if ~isempty(obj.predictionHistory)
                plot(obj.predictionHistory, 'o-', 'LineWidth', 2);
                xlabel('Frame');
                ylabel('Gesture Class');
                title('Prediction History');
                ylim([0, 7]);
                grid on;
            end
            
            % Latency history
            subplot(2, 2, 4);
            if ~isempty(obj.latencyHistory)
                plot(obj.latencyHistory, '-', 'LineWidth', 2);
                xlabel('Frame');
                ylabel('Latency (ms)');
                title('Inference Latency');
                grid on;
                
                % Add statistics
                avgLatency = mean(obj.latencyHistory);
                hold on;
                yline(avgLatency, '--r', sprintf('Avg: %.2f ms', avgLatency));
                hold off;
            end
        end
        
        function updateStatus(obj, status, color)
            % Update status display
            % Args:
            %   status: Status string
            %   color: RGB color
            
            obj.textHandles.status.String = sprintf('Status: %s', status);
            obj.textHandles.status.Color = color;
        end
        
        function stop(obj)
            % Stop the simulator
            
            obj.isRunning = false;
            obj.updateStatus('Stopped', [1, 0.5, 0]);
            
            fprintf('\nSimulator stopped\n');
            fprintf('Total frames: %d\n', obj.frameCount);
            
            if obj.frameCount > 0
                avgLatency = mean(obj.latencyHistory);
                avgFPS = obj.frameCount / toc(obj.startTime);
                fprintf('Average latency: %.2f ms\n', avgLatency);
                fprintf('Average FPS: %.1f\n', avgFPS);
            end
        end
    end
end