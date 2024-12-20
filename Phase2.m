% Load job scheduling data (jobs_data.mat contains job schedules)
load('job_schedule_data.mat'); % Assuming this file contains the jobs_data matrix
% Define threshold for makespan achievement
makespanThreshold = 50; % Example value, modify as needed
% Initialize parameters
makespan = calculateMakespan(jobs_data); % Calculate the makespan of the initial schedule
% Transition to Phase 2: TEC Minimization if makespan threshold is achieved
if makespan <= makespanThreshold
    disp('Makespan achieved. Transitioning to TEC Minimization...');
    
    % Identify schedule-critical and non-schedule-critical tasks
    [criticalTasks, nonCriticalTasks] = identifyCriticalTasks(jobs_data, makespan);
    Time_Critical_Operation();
    % Print the results of critical and non-critical tasks
    disp('Time Critical Tasks:');
    disp(criticalTasks);
    pause(1);
    disp('Non Time-Critical Tasks:');
    disp(nonCriticalTasks);
    
    % Calculate the initial TEC for the given schedule
    initialTEC = calculateTEC(jobs_data);
    bestTEC = initialTEC;
    bestSchedule = jobs_data;  % Store the best schedule found
    
    % Maximum iterations for tabu search
    maxIterations = 1000;
    tabuList = zeros(size(jobs_data));  % Tabu list to avoid recent moves
    
    % Start tabu search for TEC minimization
    for iter = 1:maxIterations
        % Generate new neighborhoods by modifying non-critical tasks
        newSchedules = generateNeighborhoods(jobs_data, nonCriticalTasks);
        
        % Print the neighborhoods generated
        disp('Generated Neighborhoods:');
        disp(newSchedules);
        pause(1);
        
        % Evaluate and select the best new schedule
        [newTEC, newSchedule] = evaluateSchedules(newSchedules);
        
        % Check if the new schedule improves TEC while maintaining the makespan
        if newTEC < bestTEC && calculateMakespan(newSchedule) <= makespan
            bestTEC = newTEC;
            bestSchedule = newSchedule;
            
            % Update tabu list with the new solution
            tabuList = updateTabuList(tabuList, newSchedule, iter);
        end
        
        % Check for convergence
        if converged(bestTEC, initialTEC)
            disp('TEC minimization converged.');
            fprintf('\n');
            break;
        end
    end
    
    % Display results
    disp('Phase 2: TEC Minimization completed.');
    pause(1);
    % Assuming jobs_data contains the scheduling data matrix with energy consumption in the appropriate column
    % Calculate and print the initial TEC
    % Initial TEC calculation using a custom function (assuming calculateTEC is implemented)
    initialTEC = calculateTEC1(jobs_data);
     % Display the initial TEC
     disp(['Initial TEC: ', num2str(initialTEC)]);
     pause(1);
     % Display the best TEC
     disp(['Best TEC: ', num2str(bestTEC)]);
     fprintf('\n');
     pause(1);
     % Calculate the difference between Best TEC and Initial TEC
     CostDifference = initialTEC - bestTEC; % Directly subtract as numbers
     disp(['Cost difference: ', num2str(CostDifference)]); % Display the cost difference
     fprintf('\n');
     pause(1);
     % Display the best schedule (assuming bestSchedule is computed somewhere)
     disp('Best Schedule:');
     disp(bestSchedule); % Display the optimized schedule
    pause(1);
else
    disp('Makespan not achieved. TEC Minimization skipped.');
end

% Function to calculate Total Energy Costs (TEC)
function TEC = calculateTEC1(jobs_data)
    TEC = sum(jobs_data(:, 1)); % Assuming column 1 contains energy consumption
end
%% Supporting Functions:

% Function to calculate makespan (max end time)
function makespan = calculateMakespan(jobs_data)
    makespan = max(jobs_data(:, 2)); % Assuming column 2 contains end times
end

% Function to identify critical and non-critical tasks based on slack time
function [criticalTasks, nonCriticalTasks] = identifyCriticalTasks(jobs_data, makespan)
    % Calculate slack (difference between latest finish time and actual finish time)
    slackTimes = jobs_data(:, 3) - jobs_data(:, 2); % Example: Slack = latest finish - actual finish time
    criticalThreshold = 0; % Tasks with no slack are critical (can modify based on your definition)
    
    criticalTasks = jobs_data(slackTimes <= criticalThreshold, :);  % Tasks that directly affect the makespan
    nonCriticalTasks = jobs_data(slackTimes > criticalThreshold, :); % Tasks with slack that are non-critical
    
    % Print the critical and non-critical tasks (added print statement)
    disp('Slack Times:');
    disp(slackTimes);
    pause(1);
end

% Function to calculate Total Energy Costs (TEC)
function TEC = calculateTEC(jobs_data)
    TEC = sum(jobs_data(:, 4)); % Assuming column 4 contains energy consumption
end

% Function to generate new neighborhoods by modifying non-critical tasks
function newSchedules = generateNeighborhoods(jobs_data, nonCriticalTasks)
    newSchedules = {};  % Store the generated neighborhoods
    
    % 1. Move non-internal tasks (tasks at the start or end) within the same machine or to another machine
    disp('Moving Non-Internal Tasks:');
    for i = 1:size(nonCriticalTasks, 1)
        % Modify job schedule by shifting non-internal tasks
        disp(['Modifying task: ', num2str(i)]);
        % Implement logic for shifting tasks within machines or across machines
    end
    
    % 2. Swap internal tasks (tasks in the middle) within the same machine
    disp('Swapping Internal Tasks:');
    pause(1);
    % Implement logic for swapping tasks within the same machine
    
    % 3. Shift tasks across machines to balance loads
    disp('Shifting Tasks Across Machines:');
    pause(1);
    % Implement logic for swapping or shifting tasks across machines
end

% Function to evaluate TEC for generated schedules
function [newTEC, newSchedule] = evaluateSchedules(newSchedules)
    newTEC = inf;
    newSchedule = [];
    for i = 1:length(newSchedules)
        TEC = calculateTEC(newSchedules{i});
        if TEC < newTEC
            newTEC = TEC;
            newSchedule = newSchedules{i};
        end
    end
end

% Function to update tabu list with new schedule
function tabuList = updateTabuList(tabuList, newSchedule, iter)
    % Update the tabu list to prevent revisiting recent moves
    tabuList(:,:,iter) = newSchedule;  % Modify as needed to track moves
end

% Function to check for convergence (change in TEC is small)
function isConverged = converged(currentTEC, previousTEC)
    isConverged = abs(currentTEC - previousTEC) < 0.01; % Example convergence threshold
end

% Define number of jobs
num_jobs = size(jobs_data, 1);

% Initialize job data structure
job_data = struct;

% Initialize time-critical operations list
time_critical_operations = [];
fileID = fopen('output.txt', 'r');
if fileID == -1
    error('Failed to open the file for reading.');
end
line = fgets(fileID);  
fclose(fileID);
value = sscanf(line, 'The value is: %d');  % Convert the string to a number
% Calculate Cmax (maximum completion time)
Cmax = 0;  % Initialize to 0
Cbar = num2str(value);  % Define based on your specific requirements
% Process the data to extract job, operation, start time, end time, and machine assignment
for i = 1:num_jobs
    job_data(i).Job = i;
    num_operations = sum(~isnan(jobs_data(i, :))); % Number of valid operations for this job
    
    for j = 1:num_operations
        start_time = sum(jobs_data(i, 1:j-1), 'omitnan'); % Cumulative start time
        end_time = start_time + jobs_data(i, j); % End time is start time + processing time
        machine_assigned = mod(i + j, 3); % Example machine assignment logic (adjust if needed)
        
        job_data(i).OperationNo(j) = j;
        job_data(i).StartTime(j) = start_time;
        job_data(i).EndTime(j) = end_time;
        job_data(i).AssignedMachine(j) = machine_assigned;
        
        % Update Cmax for time-critical operations
        Cmax = max(Cmax, end_time);  % Find the maximum end time (Cmax)
    end
end

% Calculate Cbar (define based on specific requirements, example: total processing time)
Cbar = sum(jobs_data(:), 'omitnan');  % Sum of all job durations (adjust logic if needed)

% Save to Excel
filename = 'job_scheduling.xlsx';
headers = {'Job', 'OperationNo', 'StartTime', 'EndTime', 'AssignedMachine'};

% Prepare data for saving
output_data = [];
for i = 1:num_jobs
    num_operations = sum(~isnan(job_data(i).StartTime)); % Number of valid operations for this job
    for j = 1:num_operations
        % Only add valid operations (non-NaN values)
        output_data = [output_data; job_data(i).Job, job_data(i).OperationNo(j), job_data(i).StartTime(j), job_data(i).EndTime(j), job_data(i).AssignedMachine(j)];
    end
end

% Write to Excel
writecell([headers; num2cell(output_data)], filename);

% Check for time-critical operations
for i = 1:num_jobs
    num_operations = sum(~isnan(job_data(i).StartTime)); % Number of valid operations for this job
    for j = 1:num_operations
        if ~isnan(job_data(i).StartTime(j))
            rij = job_data(i).StartTime(j);
            pk_ij = jobs_data(i, j); % Processing time
            qij = job_data(i).EndTime(j) - rij; % End time minus start time
            if rij + pk_ij + qij == Cmax || (job_data(i).EndTime(j) - job_data(i).StartTime(j)) == (Cbar - Cmax)
                time_critical_operations = [time_critical_operations; job_data(i).Job, job_data(i).OperationNo(j)];
            end
        end
    end
end

% Check the number of columns in bestSchedule
numCols = size(bestSchedule, 2);

% Define column names based on the number of columns
if numCols == 10
    columnNames = {'Job1', 'Job2', 'Job3', 'Job4', 'Job5', 'Job6', 'Job7', 'Job8', 'Job9', 'Job10'};
elseif numCols == 5
    columnNames = {'Job', 'OperationNo', 'StartTime', 'EndTime', 'AssignedMachine'};
else
    disp('Unexpected number of columns. Adjusting to default names.');
    columnNames = arrayfun(@(x) ['Var' num2str(x)], 1:numCols, 'UniformOutput', false);
end

% Replace NaN values with zeros or any other value (optional)
bestSchedule(isnan(bestSchedule)) = 0;

% Create the table with the correct number of columns
bestScheduleTable = array2table(bestSchedule, 'VariableNames', columnNames);

% Specify the file name to save the results
outputFileName = 'BestSchedule.xlsx';

% Write the table to the Excel file
writetable(bestScheduleTable, outputFileName);

pause(1);
fprintf('|-----------------|\n');
fprintf('|------ END ------|\n');
fprintf('|-----------------|\n');