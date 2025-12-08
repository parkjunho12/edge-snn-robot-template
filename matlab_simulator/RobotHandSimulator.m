% EMG Robot Hand Simulator with Full Finger Control
% Visualizes complete robot hand with individual finger movements
% Based on EMG inference predictions
%
% Features:
%   - 5 fingers (thumb, index, middle, ring, pinky)
%   - 3 joints per finger (MCP, PIP, DIP)
%   - Wrist rotation and flexion
%   - Real-time EMG control
%   - Beautiful 3D visualization
%
% Usage:
%   1. Start FastAPI server: ./start_server.sh
%   2. Run this script in MATLAB
%   3. Watch robot hand move based on EMG predictions
%
% Author: EMG Inference System v0.3
% Date: 2025

classdef RobotHandSimulator < handle
    % Complete Robot Hand Simulator with Individual Finger Control
    
    properties
        % Server connection
        serverURL = 'http://localhost:8000';
        
        % Hand configuration
        % Each finger has 3 joints: MCP (base), PIP (middle), DIP (tip)
        fingers = struct(...
            'thumb', struct('joints', [0, 0, 0], 'lengths', [0.04, 0.03, 0.025]), ...
            'index', struct('joints', [0, 0, 0], 'lengths', [0.05, 0.04, 0.03]), ...
            'middle', struct('joints', [0, 0, 0], 'lengths', [0.055, 0.045, 0.035]), ...
            'ring', struct('joints', [0, 0, 0], 'lengths', [0.05, 0.04, 0.03]), ...
            'pinky', struct('joints', [0, 0, 0], 'lengths', [0.04, 0.035, 0.025]) ...
        );
        
        % Target finger positions
        targetFingers;
        
        % Wrist parameters
        wristRotation = 0;      % Rotation around Z axis (rad)
        wristFlexion = 0;       % Flexion/extension (rad)
        targetWristRotation = 0;
        targetWristFlexion = 0;
        
        % Palm dimensions
        palmWidth = 0.08;       % Width of palm (meters)
        palmLength = 0.10;      % Length of palm (meters)
        
        % Visualization
        fig;                    % Figure handle
        ax;                     % 3D axis handle
        handPlots;             % Plot handles for each finger
        palmPlot;              % Palm plot handle
        textHandles;           % Text display handles
        historyAxes;           % History plot axes
        
        % Prediction history
        predictionHistory = [];
        confidenceHistory = [];
        latencyHistory = [];
        maxHistoryLength = 100;
        
        % Animation parameters
        animationSpeed = 0.15;  % Interpolation speed (0-1)
        updateRate = 30;        % Hz
        
        % Gesture mapping
        gestureMappings = containers.Map('KeyType', 'double', 'ValueType', 'any');
        
        % Status
        isRunning = false;
        frameCount = 0;
        startTime;
    end
    
    methods
        function obj = RobotHandSimulator(serverURL)
            % Constructor
            % Args:
            %   serverURL: URL of FastAPI server (optional)
            
            if nargin > 0
                obj.serverURL = serverURL;
            end
            
            % Initialize target fingers (copy of current state)
            obj.targetFingers = obj.fingers;
            
            % Initialize visualization
            obj.initializeVisualization();
            
            % Initialize gesture mappings
            obj.initializeGestureMappings();
            
            % Test server connection
            obj.testConnection();
        end
        
        function initializeGestureMappings(obj)
            % Initialize gesture mappings with detailed finger positions
            
            % Gesture 0: Rest (all fingers relaxed)
            obj.gestureMappings(0) = struct(...
                'name', 'Rest', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.0, ...
                'thumb', [0.0, 0.0, 0.0], ...
                'index', [0.0, 0.0, 0.0], ...
                'middle', [0.0, 0.0, 0.0], ...
                'ring', [0.0, 0.0, 0.0], ...
                'pinky', [0.0, 0.0, 0.0], ...
                'color', [0.5, 0.5, 0.5]);
            
            % Gesture 1: Hand Open (all fingers extended)
            obj.gestureMappings(1) = struct(...
                'name', 'Hand Open', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.0, ...
                'thumb', [0.8, 0.0, 0.0], ...    % Thumb abducted
                'index', [0.0, 0.0, 0.0], ...    % Straight
                'middle', [0.0, 0.0, 0.0], ...   % Straight
                'ring', [0.0, 0.0, 0.0], ...     % Straight
                'pinky', [0.0, 0.0, 0.0], ...    % Straight
                'color', [0.0, 1.0, 0.0]);
            
            % Gesture 2: Hand Close/Fist (all fingers flexed)
            obj.gestureMappings(2) = struct(...
                'name', 'Hand Close', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.0, ...
                'thumb', [0.3, 1.2, 1.0], ...    % Thumb wrapped
                'index', [1.0, 1.2, 1.0], ...    % Fully flexed
                'middle', [1.0, 1.2, 1.0], ...   % Fully flexed
                'ring', [1.0, 1.2, 1.0], ...     % Fully flexed
                'pinky', [1.0, 1.2, 1.0], ...    % Fully flexed
                'color', [1.0, 0.0, 0.0]);
            
            % Gesture 3: Wrist Flex (hand bent upward)
            obj.gestureMappings(3) = struct(...
                'name', 'Wrist Flex', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.5, ...         % Flexed up
                'thumb', [0.3, 0.2, 0.1], ...
                'index', [0.2, 0.2, 0.1], ...
                'middle', [0.2, 0.2, 0.1], ...
                'ring', [0.2, 0.2, 0.1], ...
                'pinky', [0.2, 0.2, 0.1], ...
                'color', [0.0, 0.0, 1.0]);
            
            % Gesture 4: Wrist Extend (hand bent downward)
            obj.gestureMappings(4) = struct(...
                'name', 'Wrist Extend', ...
                'wristRotation', 0.0, ...
                'wristFlexion', -0.5, ...        % Extended down
                'thumb', [0.3, 0.2, 0.1], ...
                'index', [0.2, 0.2, 0.1], ...
                'middle', [0.2, 0.2, 0.1], ...
                'ring', [0.2, 0.2, 0.1], ...
                'pinky', [0.2, 0.2, 0.1], ...
                'color', [1.0, 1.0, 0.0]);
            
            % Gesture 5: Pinch (thumb and index touching)
            obj.gestureMappings(5) = struct(...
                'name', 'Pinch', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.0, ...
                'thumb', [0.5, 0.8, 0.6], ...    % Thumb flexed
                'index', [0.8, 1.0, 0.8], ...    % Index flexed
                'middle', [0.2, 0.3, 0.2], ...   % Slightly flexed
                'ring', [0.2, 0.3, 0.2], ...     % Slightly flexed
                'pinky', [0.2, 0.3, 0.2], ...    % Slightly flexed
                'color', [1.0, 0.0, 1.0]);
            
            % Gesture 6: Point (index extended, others flexed)
            obj.gestureMappings(6) = struct(...
                'name', 'Point', ...
                'wristRotation', 0.0, ...
                'wristFlexion', 0.0, ...
                'thumb', [0.3, 1.0, 0.8], ...    % Thumb folded
                'index', [0.0, 0.0, 0.0], ...    % Index extended
                'middle', [1.0, 1.2, 1.0], ...   % Middle flexed
                'ring', [1.0, 1.2, 1.0], ...     % Ring flexed
                'pinky', [1.0, 1.2, 1.0], ...    % Pinky flexed
                'color', [0.0, 1.0, 1.0]);
        end
        
        function initializeVisualization(obj)
            % Initialize 3D visualization of robot hand
            
            obj.fig = figure('Name', 'EMG Robot Hand Simulator', ...
                           'NumberTitle', 'off', ...
                           'Position', [100, 100, 1400, 900], ...
                           'Color', [0.95, 0.95, 0.95]);
            
            % Create main 3D plot (larger)
            obj.ax = subplot(2, 3, [1, 2, 4, 5]);
            hold(obj.ax, 'on');
            grid(obj.ax, 'on');
            axis(obj.ax, 'equal');
            xlabel(obj.ax, 'X (m)');
            ylabel(obj.ax, 'Y (m)');
            zlabel(obj.ax, 'Z (m)');
            title(obj.ax, 'Robot Hand - EMG Control', 'FontSize', 14, 'FontWeight', 'bold');
            view(obj.ax, 45, 30);
            
            % Set axis limits
            limit = 0.15;
            xlim(obj.ax, [-limit, limit]);
            ylim(obj.ax, [-limit, limit]);
            zlim(obj.ax, [-0.05, limit*2]);
            
            % Initialize hand plots
            obj.handPlots = struct();
            fingerNames = {'thumb', 'index', 'middle', 'ring', 'pinky'};
            colors = [
                0.8, 0.4, 0.4;  % Thumb: red-ish
                0.4, 0.6, 0.8;  % Index: blue-ish
                0.4, 0.8, 0.6;  % Middle: green-ish
                0.8, 0.6, 0.4;  % Ring: orange-ish
                0.6, 0.4, 0.8   % Pinky: purple-ish
            ];
            
            for i = 1:length(fingerNames)
                obj.handPlots.(fingerNames{i}) = plot3(obj.ax, 0, 0, 0, 'o-', ...
                    'LineWidth', 3, ...
                    'MarkerSize', 8, ...
                    'Color', colors(i, :), ...
                    'MarkerFaceColor', colors(i, :));
            end
            
            % Initialize palm plot
            obj.palmPlot = patch(obj.ax, 'XData', [], 'YData', [], 'ZData', [], ...
                'FaceColor', [0.9, 0.75, 0.6], ...
                'FaceAlpha', 0.8, ...
                'EdgeColor', [0.5, 0.4, 0.3], ...
                'LineWidth', 2);
            
            % Create prediction history plot
            obj.historyAxes.prediction = subplot(2, 3, 3);
            title('Prediction History');
            xlabel('Frame');
            ylabel('Gesture');
            ylim([0, 7]);
            yticks(0:6);
            yticklabels({'Rest', 'Open', 'Close', 'Flex', 'Extend', 'Pinch', 'Point'});
            grid on;
            
            % Create latency plot
            obj.historyAxes.latency = subplot(2, 3, 6);
            title('Inference Latency');
            xlabel('Frame');
            ylabel('Latency (ms)');
            grid on;
            
            % Add text displays
            obj.textHandles = struct();
            
            obj.textHandles.status = annotation('textbox', [0.02, 0.96, 0.3, 0.03], ...
                'String', 'Status: Initializing...', ...
                'EdgeColor', 'none', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'Color', [0, 0, 0]);
            
            obj.textHandles.gesture = annotation('textbox', [0.02, 0.92, 0.3, 0.03], ...
                'String', 'Gesture: None', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.confidence = annotation('textbox', [0.02, 0.88, 0.3, 0.03], ...
                'String', 'Confidence: 0%', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.fps = annotation('textbox', [0.02, 0.84, 0.15, 0.03], ...
                'String', 'FPS: 0', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            obj.textHandles.latency = annotation('textbox', [0.17, 0.84, 0.15, 0.03], ...
                'String', 'Latency: 0 ms', ...
                'EdgeColor', 'none', ...
                'FontSize', 11);
            
            % Add finger labels
            obj.textHandles.fingerLabels = annotation('textbox', [0.02, 0.78, 0.3, 0.05], ...
                'String', sprintf('Thumb | Index | Middle | Ring | Pinky\n[0,0,0]  [0,0,0]  [0,0,0]  [0,0,0]  [0,0,0]'), ...
                'EdgeColor', [0.7, 0.7, 0.7], ...
                'FontSize', 9, ...
                'FontName', 'Courier');
            
            drawnow;
        end
        
        function success = testConnection(obj)
            % Test connection to FastAPI server
            
            try
                url = sprintf('%s/health', obj.serverURL);
                response = webread(url);
                
                if strcmp(response.status, 'ok')
                    fprintf('✓ Connected to server at %s\n', obj.serverURL);
                    obj.updateStatus('Connected', [0, 0.6, 0]);
                    success = true;
                else
                    warning('Server returned unexpected status');
                    success = false;
                end
            catch ME
                warning('Failed to connect to server: %s', ME.message);
                obj.updateStatus('Disconnected', [0.8, 0, 0]);
                success = false;
            end
        end
        
        function start(obj, duration)
            % Start streaming inference and visualization
            
            if nargin < 2
                duration = inf;
            end
            
            obj.isRunning = true;
            obj.frameCount = 0;
            obj.startTime = tic;
            
            fprintf('========================================\n');
            fprintf('EMG Robot Hand Simulator Started\n');
            fprintf('========================================\n');
            fprintf('Duration: %.1f seconds\n', duration);
            fprintf('Update Rate: %d Hz\n', obj.updateRate);
            fprintf('Press Ctrl+C to stop\n\n');
            
            obj.updateStatus('Running', [0, 0.8, 0]);
            
            % Start polling loop
            obj.pollingLoop(duration);
        end
        
        function pollingLoop(obj, duration)
            % Polling-based inference loop
            
            frameInterval = 5.0; 
            
            while obj.isRunning && toc(obj.startTime) < duration
                frameStart = tic;
                
                try
                    % Call inference endpoint
                    url = sprintf('%s/infer', obj.serverURL);
                    options = weboptions('ContentType', 'json', 'Timeout', 5);
                    
                    request = struct('encoding_type', 'rate', ...
                                   'model_prefix', 'tcn', ...
                                   'device', 'cpu');
                    
                    response = webwrite(url, request, options);
                    
                    % Parse response
                    prediction = str2double(response.pred_idx);
                    confidence = str2double(response.conf);
                    latency = str2double(response.latency_ms);
                    
                    % Update hand
                    obj.updateHand(prediction, confidence, latency);
                    
                    obj.frameCount = obj.frameCount + 1;
                    
                catch ME
                    fprintf('Frame %d error: %s\n', obj.frameCount, ME.message);
                end
                
                % Maintain frame rate
                frameElapsed = toc(frameStart);
                if frameElapsed < frameInterval
                    pause(frameInterval - frameElapsed);
                end
                
                % Allow MATLAB to process events
                drawnow;
                
                % Check for figure close
                if ~ishandle(obj.fig)
                    obj.isRunning = false;
                    break;
                end
            end
            
            obj.stop();
        end
        
        function updateHand(obj, prediction, confidence, latency)
            % Update hand visualization based on prediction
            
            % Update history
            obj.predictionHistory(end+1) = prediction;
            obj.confidenceHistory(end+1) = confidence;
            obj.latencyHistory(end+1) = latency;
            
            if length(obj.predictionHistory) > obj.maxHistoryLength
                obj.predictionHistory(1) = [];
                obj.confidenceHistory(1) = [];
                obj.latencyHistory(1) = [];
            end
            
            % Get gesture mapping
            if obj.gestureMappings.isKey(prediction)
                gesture = obj.gestureMappings(prediction);
                
                % Update target positions
                obj.targetWristRotation = gesture.wristRotation;
                obj.targetWristFlexion = gesture.wristFlexion;
                obj.targetFingers.thumb.joints = gesture.thumb;
                obj.targetFingers.index.joints = gesture.index;
                obj.targetFingers.middle.joints = gesture.middle;
                obj.targetFingers.ring.joints = gesture.ring;
                obj.targetFingers.pinky.joints = gesture.pinky;
                
                gestureName = gesture.name;
                gestureColor = gesture.color;
            else
                gestureName = 'Unknown';
                gestureColor = [0.5, 0.5, 0.5];
            end
            
            % Smoothly interpolate to target positions
            obj.wristRotation = obj.wristRotation + ...
                obj.animationSpeed * (obj.targetWristRotation - obj.wristRotation);
            obj.wristFlexion = obj.wristFlexion + ...
                obj.animationSpeed * (obj.targetWristFlexion - obj.wristFlexion);
            
            fingerNames = fieldnames(obj.fingers);
            for i = 1:length(fingerNames)
                name = fingerNames{i};
                obj.fingers.(name).joints = obj.fingers.(name).joints + ...
                    obj.animationSpeed * (obj.targetFingers.(name).joints - obj.fingers.(name).joints);
            end
            
            % Update 3D visualization
            obj.updateVisualization();
            
            % Update text displays
            obj.textHandles.gesture.String = sprintf('Gesture: %s (%d)', gestureName, prediction);
            obj.textHandles.confidence.String = sprintf('Confidence: %.1f%%', confidence * 100);
            obj.textHandles.latency.String = sprintf('Latency: %.2f ms', latency);
            
            % Update finger angles display
            fingerStr = sprintf('Thumb | Index | Middle | Ring  | Pinky\n');
            fingerStr = [fingerStr sprintf('[%.1f,%.1f,%.1f]', obj.fingers.thumb.joints)];
            fingerStr = [fingerStr sprintf(' [%.1f,%.1f,%.1f]', obj.fingers.index.joints)];
            fingerStr = [fingerStr sprintf(' [%.1f,%.1f,%.1f]', obj.fingers.middle.joints)];
            fingerStr = [fingerStr sprintf(' [%.1f,%.1f,%.1f]', obj.fingers.ring.joints)];
            fingerStr = [fingerStr sprintf(' [%.1f,%.1f,%.1f]', obj.fingers.pinky.joints)];
            obj.textHandles.fingerLabels.String = fingerStr;
            
            % Calculate FPS
            if obj.frameCount > 0
                elapsedTime = toc(obj.startTime);
                fps = obj.frameCount / elapsedTime;
                obj.textHandles.fps.String = sprintf('FPS: %.1f', fps);
            end
            
            % Update history plots
            obj.updateHistoryPlots();
        end
        
        function updateVisualization(obj)
            % Update 3D hand visualization
            
            % Wrist transformation
            Twrist = makehgtform('zrotate', obj.wristRotation, ...
                               'yrotate', obj.wristFlexion);
            
            % Draw palm
            obj.drawPalm(Twrist);
            
            % Draw each finger
            fingerNames = {'thumb', 'index', 'middle', 'ring', 'pinky'};
            fingerOffsets = [
                -obj.palmWidth*0.4, obj.palmLength*0.3, 0, pi/4;    % Thumb
                -obj.palmWidth*0.3, obj.palmLength*0.5, 0, 0;      % Index
                0, obj.palmLength*0.5, 0, 0;                       % Middle
                obj.palmWidth*0.3, obj.palmLength*0.5, 0, 0;       % Ring
                obj.palmWidth*0.4, obj.palmLength*0.3, 0, -pi/6    % Pinky
            ];
            
            for i = 1:length(fingerNames)
                name = fingerNames{i};
                offset = fingerOffsets(i, :);
                obj.drawFinger(name, Twrist, offset);
            end
        end
        
        function drawPalm(obj, Twrist)
            % Draw palm as a rectangular patch
            
            w = obj.palmWidth / 2;
            l = obj.palmLength;
            
            % Palm vertices (local coordinates)
            vertices = [
                -w, 0, 0;
                 w, 0, 0;
                 w, l, 0;
                -w, l, 0
            ];
            
            % Transform to world coordinates
            worldVertices = zeros(size(vertices));
            for i = 1:size(vertices, 1)
                v = [vertices(i, :), 1]';
                v_world = Twrist * v;
                worldVertices(i, :) = v_world(1:3)';
            end
            
            % Update palm patch
            set(obj.palmPlot, ...
                'XData', worldVertices(:, 1), ...
                'YData', worldVertices(:, 2), ...
                'ZData', worldVertices(:, 3));
        end
        
        function drawFinger(obj, fingerName, Twrist, offset)
            % Draw a single finger with all joints
            
            finger = obj.fingers.(fingerName);
            joints = finger.joints;
            lengths = finger.lengths;
            
            % Start from palm offset
            Tlocal = makehgtform('translate', [offset(1), offset(2), offset(3)], ...
                               'zrotate', offset(4));
            T = Twrist * Tlocal;
            
            % Store joint positions
            positions = zeros(4, 3);
            positions(1, :) = T(1:3, 4)';  % Base
            
            % Each joint
            for i = 1:3
                % Rotate around Y axis (flexion/extension)
                Tjoint = makehgtform('yrotate', joints(i));
                % Translate along X (finger length)
                Ttrans = makehgtform('translate', [lengths(i), 0, 0]);
                
                T = T * Tjoint * Ttrans;
                positions(i+1, :) = T(1:3, 4)';
            end
            
            % Update finger plot
            set(obj.handPlots.(fingerName), ...
                'XData', positions(:, 1), ...
                'YData', positions(:, 2), ...
                'ZData', positions(:, 3));
        end
        
        function updateHistoryPlots(obj)
            % Update prediction and latency history plots
            
            % Prediction history
            axes(obj.historyAxes.prediction);
            cla;
            if ~isempty(obj.predictionHistory)
                plot(obj.predictionHistory, 'o-', 'LineWidth', 2, 'MarkerSize', 4);
                xlabel('Frame');
                ylabel('Gesture');
                title('Prediction History');
                ylim([0, 7]);
                yticks(0:6);
                yticklabels({'Rest', 'Open', 'Close', 'Flex', 'Extend', 'Pinch', 'Point'});
                grid on;
            end
            
            % Latency history
            axes(obj.historyAxes.latency);
            cla;
            if ~isempty(obj.latencyHistory)
                plot(obj.latencyHistory, '-', 'LineWidth', 2);
                xlabel('Frame');
                ylabel('Latency (ms)');
                title('Inference Latency');
                grid on;
                
                % Add statistics
                if length(obj.latencyHistory) > 10
                    avgLatency = mean(obj.latencyHistory);
                    hold on;
                    yline(avgLatency, '--r', sprintf('Avg: %.2f ms', avgLatency), 'LineWidth', 1.5);
                    hold off;
                end
            end
        end
        
        function updateStatus(obj, status, color)
            % Update status display
            
            obj.textHandles.status.String = sprintf('Status: %s', status);
            obj.textHandles.status.Color = color;
        end
        
        function stop(obj)
            % Stop the simulator
            
            obj.isRunning = false;
            obj.updateStatus('Stopped', [0.8, 0.4, 0]);
            
            fprintf('\n========================================\n');
            fprintf('Simulator Stopped\n');
            fprintf('========================================\n');
            fprintf('Total frames: %d\n', obj.frameCount);
            
            if obj.frameCount > 0
                avgLatency = mean(obj.latencyHistory);
                avgFPS = obj.frameCount / toc(obj.startTime);
                fprintf('Average latency: %.2f ms\n', avgLatency);
                fprintf('Average FPS: %.1f\n', avgFPS);
                fprintf('Total time: %.1f seconds\n', toc(obj.startTime));
            end
        end
    end
end