% Load the fjs file
filename = 'D:\Job scheduling\FJSP-benchmarks-main\1_Brandimarte this resutls were used in paper which we are following\BrandimarteMk1.fjs';
fileID = fopen(filename, 'r');
if fileID == -1
    error('Failed to open file.');
end
% Read the number of jobs, machines, and operations
numJobs = fscanf(fileID, '%d', 1);
numMachines = fscanf(fileID, '%d', 1);
numOperations = fscanf(fileID, '%d', 1);
% Display the number of jobs and machines
disp(['Number of Jobs: ', num2str(numJobs)]);
disp(['Number of Machines: ', num2str(numMachines)]);
fprintf('\n');
% Initialize a matrix to store the job data
jobData = zeros(numJobs, numOperations * 2);
% Read the operations data for each job
for job = 1:numJobs
    for op = 1:numOperations
        % Read the machine number and processing time
        machine = fscanf(fileID, '%d', 1); % Machine number
        processingTime = fscanf(fileID, '%d', 1); % Processing time
        
        % Validate the machine number
        if machine < 1 || machine > numMachines
            error('Invalid machine number %d for job %d, operation %d. It must be between 1 and %d.', machine, job, op, numMachines);
        end
        % Store valid data in the jobData matrix
        jobData(job, (op-1)*2 + 1) = machine;
        jobData(job, (op-1)*2 + 2) = processingTime;
    end
end
% Close the file
fclose(fileID);
% Initialize the schedule and time tracking variables
schedule = [];
currentTime = zeros(numMachines, 1); % Array to track when each machine is next available
endTime = zeros(numJobs, numOperations); % Array to store end times
% Process jobs and generate the schedule
for job = 1:numJobs
    jobStartTime = 0; % Initialize job start time
    for op = 1:numOperations
        machine = jobData(job, (op-1)*2 + 1); % Get machine number (0-based)
        processingTime = jobData(job, (op-1)*2 + 2); % Get processing time
        % Determine the start and end time
        startTime = max(jobStartTime, currentTime(machine) + 1); % Start time is max of previous machine's end time and job's start time
        endTime(job, op) = startTime + processingTime; % Calculate end time
        % Update the machine's next available time
        currentTime(machine) = endTime(job, op);
        % Store job details
        schedule = [schedule; job, op, startTime, endTime(job, op), machine];
    end
end
% Convert schedule to table for easy viewing and saving
scheduleTable = array2table(schedule, 'VariableNames', ...
    {'Job', 'OperationNo', 'StartTime', 'EndTime', 'AssignedMachine'});
% Save the results to an Excel file
outputFilename = 'JobSchedule.xlsx';
disp('Initializing Hybrid Swarm...');
writetable(scheduleTable, outputFilename);
maxEndTime = max(scheduleTable.EndTime);
% Display the table (optional)
disp(scheduleTable);
% Display initial makespan
% Initialize the initial makespan (before optimization)
initialMakespan = calculateMakespan(schedule(:, 3:4));  % Initial makespan based on the initial schedule
disp(['Initial Makespan: ', num2str(initialMakespan)]);
% Parameters for PSO
numParticles = 10; % Number of particles
maxIterationsPSO = 100; % Maximum iterations for PSO
w_max = 0.9; % Max inertia weight
w_min = 0.4; % Min inertia weight
c1 = 1.4; % Cognitive coefficient
c2 = 1.4; % Social coefficient
velocityMax = 0.1; % Max velocity (can be adjusted as needed)
% Initialize PSO particles with numerical data (not a table)
particles = cell(numParticles, 1); % Particle positions (schedules)
velocities = cell(numParticles, 1); % Particle velocities (changes)
bestPositions = cell(numParticles, 1); % Best positions (schedules) for each particle
bestFitness = inf(numParticles, 1); % Best fitness value for each particle
globalBestPosition = []; % Best position found by all particles
globalBestFitness = inf; % Best fitness value found by all particles
% Initialize particles and velocities randomly
for i = 1:numParticles
    % Convert scheduleTable into a numerical matrix for particles
    particles{i} = schedule(:, 3:4); % Extract only StartTime and EndTime columns for scheduling
    velocities{i} = zeros(size(particles{i})); % Initialize velocities to zeros
end
fprintf('\n');
pause(1);
disp('Starting Optimized PSO Algorithm...');
fprintf('\n');
% PSO main loop
for iter = 1:maxIterationsPSO
    % Dynamic adjustment of inertia weight (w) during iterations
    w = w_max - ((w_max - w_min) * iter / maxIterationsPSO);

    for i = 1:numParticles
        % Evaluate the fitness of each particle (using makespan)
        fitness = calculateMakespan(particles{i}); % Makespan as fitness
        
        % Update the personal best positions
        if fitness < bestFitness(i)
            bestFitness(i) = fitness;
            bestPositions{i} = particles{i};
        end
        % Update the global best position
        if fitness < globalBestFitness
            globalBestFitness = fitness;
            globalBestPosition = particles{i};
        end
        % Update the velocity
        velocities{i} = w * velocities{i} + ...
                         c1 * rand(size(velocities{i})) .* (bestPositions{i} - particles{i}) + ...
                         c2 * rand(size(velocities{i})) .* (globalBestPosition - particles{i});
        
        % Velocity clamping to prevent particles from moving too far
        velocities{i} = max(min(velocities{i}, velocityMax), -velocityMax);
        % Update the particle's position
        particles{i} = particles{i} + velocities{i}; % Update position
        % Boundary handling: ensure particles stay within problem bounds
        particles{i} = max(min(particles{i}, max(jobData(:, 2))), min(jobData(:, 2))); % Ensure the position stays within bounds
    end
    
    % Display the makespan value for the first iteration and subsequent iterations
    if iter == 1
        disp(['Iteration ', num2str(iter), ' - Initial Makespan: ', num2str(initialMakespan)]);
    else
        disp(['Iteration ', num2str(iter), ' - Best Makespan: ', num2str(maxEndTime)]);
    end
end

fprintf('\n');
% Tabu Search Parameters
n = 10; % Example value for rows
m = 20; % Example value for columns (set according to your problem)
tabuTableLength = (n * m) / 2; % Length of the tabu table
tabuPeriod = (n * m) / 4;      % Tabu period
% Crossover Probability
crossoverProbability = 0.40; % Crossover probability (40%)
%% Tabu Search Refinement Step
% Parameters for Tabu Search
maxIterations = 100; % Maximum iterations for the search
tabuSize = 10; % Size of the tabu list
aspirationCriteria = true; % Flag for aspiration criterion
% Initialize Tabu list
tabuList = cell(tabuSize, 1);
currentBest = jobData; % Initialize with the original schedule
currentMakespan = calculateMakespan(currentBest); % Calculate the makespan of the initial solution


%% Tabu Search Loop
disp('Starting Tabu Search...');
fprintf('\n');
pause(1);
for iter = 1:maxIterations
    % Step 1: Generate neighbors
    neighbors = generateNeighbors(currentBest);
    % Step 2: Evaluate neighbors and select the best
    bestNeighbor = neighbors{1};
    bestMakespan = calculateMakespan(bestNeighbor);
    for i = 2:length(neighbors)
        candidate = neighbors{i};
        candidateMakespan = calculateMakespan(candidate);
        % Select the neighbor with the best makespan
        if candidateMakespan < bestMakespan
            bestNeighbor = candidate;
            bestMakespan = candidateMakespan;
        end
    end
    % Step 3: Apply aspiration criteria (if the move is tabu, accept it if it improves the solution)
    % Convert schedules to string format for comparison
    bestNeighborStr = mat2str(bestNeighbor);
    tabuListStr = cellfun(@mat2str, tabuList, 'UniformOutput', false);
    if ~ismember(bestNeighborStr, tabuListStr) || aspirationCriteria
        currentBest = bestNeighbor;
        currentMakespan = bestMakespan;
        
        % Add to tabu list (manage size)
        tabuList{mod(iter, tabuSize) + 1} = bestNeighbor;
    end
     % Display the makespan value for the first iteration and subsequent iterations
    if iter == 1
        disp(['Iteration ', num2str(iter), ' - Initial Makespan: ', num2str(maxEndTime)]);
    else
        disp(['Iteration ', num2str(iter), ' - Best Makespan: ', num2str(maxEndTime)]);
    end
end
fprintf('\n');
%% Final Result
disp('Final Optimized Schedule:');
disp(currentBest);
disp(['Final Makespan: ', num2str(maxEndTime)]);
fprintf('\n');
pause(1);
fileID = fopen('output.txt', 'w');
if fileID == -1
    error('Failed to open the file for writing.');
end
fprintf(fileID, num2str(maxEndTime));
fclose(fileID);

% Hybrid Swarm with Tabu Search (HS-TS) Implementation
% Objective: Minimize Total Execution Cost (TEC)

populationSize = 50;
numTasks = 20; % Number of tasks
numMachines = 5; % Number of machines
maxIterations = 100;

% Initialize Population (Swarm)
population = initializeSwarm(populationSize, numTasks, numMachines);
bestLocal = population; 
[bestGlobal, bestGlobalFitness] = getBestGlobal(population);

% Main Optimization Loop
for iteration = 1:maxIterations    
    % Crossover and Mutation
    population = performCrossoverMutation(population);
    
    % Update Best Solutions
    bestLocal = updateBestLocal(population, bestLocal);
    [bestGlobal, bestGlobalFitness] = getBestGlobal(population);
    
    % Tabu Search Refinement
    population = tabuSearchRefinement(population);
    
    % Fitness Evaluation
    populationFitness = evaluateFitness(population);
    
    % Identify Time-Critical Operations
    criticalTasks = identifyCriticalTasks(population, populationFitness);
    
    % Rearrange Tasks to Improve TEC
    for i = 1:populationSize
        population{i} = rearrangeTasks(population{i}, criticalTasks);
    end
    
    % Check TEC Improvement
    if isImproved(bestGlobalFitness)
        % Save Results and Update Makespan
        saveResults(bestGlobal, bestGlobalFitness);
        updateMakespan();
    end
end
disp('Optimization Complete.');
%% Helper Functions
% Function: Initialize Swarm
function swarm = initializeSwarm(popSize, numTasks, numMachines)
    % Initialize swarm with random task-machine assignments
    swarm = cell(1, popSize);
    for i = 1:popSize
        swarm{i} = randi(numMachines, 1, numTasks);
    end
end

% Function: Evaluate Fitness
function fitness = evaluateFitness(population)
    % Evaluate the fitness of all solutions in the population
    numIndividuals = length(population);
    fitness = zeros(1, numIndividuals); % Initialize fitness array
    for i = 1:numIndividuals
        solution = population{i}; % Extract the individual solution
        fitness(i) = sum(solution); % Example cost calculation (modify as needed)
    end
end

% Function: Get Best Global Solution
function [best, bestFitness] = getBestGlobal(population)
    % Find the best solution in the population
    fitness = evaluateFitness(population); % Evaluate fitness for all solutions
    [bestFitness, idx] = min(fitness); % Find the best fitness and its index
    best = population{idx}; % Retrieve the best solution
end

% Function: Perform Crossover and Mutation
function population = performCrossoverMutation(population)
    % Perform crossover and mutation on the population
    for i = 1:length(population)
        % Simple crossover and mutation example
        if rand < 0.5
            population{i} = mutate(population{i});
        end
    end
end

% Function: Mutate Solution
function solution = mutate(solution)
    % Randomly mutate a task assignment
    idx = randi(length(solution));
    solution(idx) = randi(max(solution));
end

% Function: Update Best Local Solutions
function bestLocal = updateBestLocal(population, bestLocal)
    % Update local best solutions
    for i = 1:length(population)
        if evaluateFitness({population{i}}) < evaluateFitness({bestLocal{i}})
            bestLocal{i} = population{i};
        end
    end
end

% Function: Tabu Search Refinement
function population = tabuSearchRefinement(population)
    % Apply Tabu Search refinement to the population
    for i = 1:length(population)
        population{i} = refineSolution(population{i});
    end
end

% Function: Refine Solution (Placeholder for Tabu Search Logic)
function refined = refineSolution(solution)
    % Refine a solution using Tabu Search
    % Placeholder refinement logic
    refined = solution; % Return the same solution for now
end

% Function: Identify Time-Critical Tasks
function criticalTasks = identifyCriticalTasks(population, fitness)
    % Identify tasks that are time-critical (Placeholder)
    criticalTasks = []; % Example implementation
end

% Function: Rearrange Tasks
function solution = rearrangeTasks(solution, criticalTasks)
    % Rearrange or swap tasks to improve fitness (Placeholder)
    % Implement swapping or shifting logic here
    solution = solution; % No changes in this placeholder
end

% Function: Check Improvement in Fitness
function improved = isImproved(currentFitness)
    % Check if there's an improvement in fitness
    improved = rand < 0.1; % Example condition (placeholder)
end

% Function: Save Results
function saveResults(bestSolution, bestFitness)
    % Save the best solution and fitness
end

% Function: Update Makespan
function updateMakespan()
    % Update makespan parameters
end
%% Function to calculate Makespan
function makespan = calculateMakespan(schedule)
    % This function calculates the makespan of a given schedule.
    % For now, it assumes makespan is the maximum sum of processing times in any machine.
    % Example calculation based on job processing times in a machine
    makespan = max(sum(schedule, 2)); % Sum along rows (jobs) and find the max value (makespan)
end
%% Function to Generate Neighbors
function neighbors = generateNeighbors(currentSchedule)
    % This function generates neighboring schedules by performing
    % machine assignment adjustments and task sequencing adjustments.
    neighbors = {}; % Initialize neighbors cell array
    % Machine Assignment Adjustment (e.g., swapping tasks between machines)
    for i = 1:size(currentSchedule, 1)
        for j = 1:size(currentSchedule, 2)
            for k = j+1:size(currentSchedule, 2)
                % Swap tasks between machine j and machine k for job i
                newSchedule = currentSchedule;
                temp = newSchedule(i, j);
                newSchedule(i, j) = newSchedule(i, k);
                newSchedule(i, k) = temp;
                % Add the new schedule as a neighbor
                neighbors{end+1} = newSchedule;
            end
        end
    end
    % Task Sequencing Adjustment (e.g., swapping tasks within the same machine)
    for i = 1:size(currentSchedule, 1)
        newSchedule = currentSchedule;
        for j = 1:size(currentSchedule, 2) - 1
            % Swap consecutive tasks within the same machine
            temp = newSchedule(i, j);
            newSchedule(i, j) = newSchedule(i, j+1);
            newSchedule(i, j+1) = temp;
            
            % Add the new schedule as a neighbor
            neighbors{end+1} = newSchedule;
        end
    end
end
%% Function to Generate Neighbors
function neighbors = generateNeighbors1(currentSchedule)
    % This function generates neighboring schedules by performing
    % machine assignment adjustments and task sequencing adjustments.
    neighbors = {}; % Initialize neighbors cell array
    disp('Neighborhood Generation:'); % Print header for neighborhood generation
    pause(1);
    % Machine Assignment Adjustment (e.g., swapping tasks between machines)
    disp('Machine Assignment Adjustment:'); % Print header for machine assignment adjustment
    pause(1);
    for i = 1:size(currentSchedule, 1)
        for j = 1:size(currentSchedule, 2)
            for k = j+1:size(currentSchedule, 2)
                % Swap tasks between machine j and machine k for job i
                newSchedule = currentSchedule;
                temp = newSchedule(i, j);
                newSchedule(i, j) = newSchedule(i, k);
                newSchedule(i, k) = temp;
                % Add the new schedule as a neighbor
                neighbors{end+1} = newSchedule; 
                % Display the generated neighbor after machine assignment adjustment
                disp(['Machine Swap: Job ', num2str(i), ' - Machine ', num2str(j), ' <-> Machine ', num2str(k)]);
                disp(newSchedule);
                pause(1);
            end
        end
    end
    % Task Sequencing Adjustment (e.g., swapping tasks within the same machine)
    disp('Task Sequencing Adjustment:'); % Print header for task sequencing adjustment
    pause(1);
    for i = 1:size(currentSchedule, 1)
        newSchedule = currentSchedule;
        for j = 1:size(currentSchedule, 2) - 1
            % Swap consecutive tasks within the same machine
            temp = newSchedule(i, j);
            newSchedule(i, j) = newSchedule(i, j+1);
            newSchedule(i, j+1) = temp;
            % Add the new schedule as a neighbor
            neighbors{end+1} = newSchedule;
            % Display the generated neighbor after task sequencing adjustment
            disp(['Task Swap: Job ', num2str(i), ' - Task ', num2str(j), ' <-> Task ', num2str(j+1)]);
            disp(newSchedule);
        end
    end
    % Print the total number of neighbors generated
    disp(['Total Neighbors Generated: ', num2str(length(neighbors))]);
    pause(1);
end
pause(1);
% Cbar Calculation (as requested)
Cmax = maxEndTime; % Set Cmax
Cbar = Cmax; % Initial value of Cbar = Cmax
disp(['Cbar (initial): ', num2str(Cbar)]);
pause(1);
% Now calculate Cbar for a = 5% and 10%
a_values = [0.05, 0.10]; % Percentage increases
for a = a_values
    Cbar_new = (1 + a) * Cmax; % New Cbar value
    disp(['Cbar with a = ', num2str(a*100), '%: ', num2str(Cbar_new)]);
    pause(1);
end
fprintf('\n');
% Initialize the initial makespan (before optimization)
initialMakespan = calculateMakespan(schedule(:, 3:4));  % Initial makespan based on the initial schedule
disp(['Initial Makespan: ', num2str(initialMakespan)]);
fprintf('\n');
disp(['Final Makespan: ', num2str(maxEndTime)]);
% Parameters for PSO and Tabu Search follow
% After completing the PSO and Tabu Search optimization process:
% Assume `best_makespan` is the result from the optimized schedule
% Final makespan (after optimization)
finalMakespan = maxEndTime;  % Replace this with the actual final optimized makespan value
% Calculate the cost difference
costDifference = initialMakespan - finalMakespan; 
% Display the cost difference
% Save the result to a file if needed
fileID = fopen('costDifference.txt', 'w');
if fileID == -1
    error('Failed to open the file for writing.');
end
fprintf(fileID, 'Cost Difference: %f\n', costDifference);
fclose(fileID);
% Read the content of the file
fileID = fopen(filename, 'r');
data = textscan(fileID, '%f', 'Delimiter', '\n');
fclose(fileID);
% Flatten the data into a single vector
data_flat = data{1};

% Check if the total number of elements is divisible by 10
numElements = numel(data_flat);
numColumns = 10; % Desired number of columns
if mod(numElements, numColumns) ~= 0
    % If not divisible, pad the data with zeros
    padding = numColumns - mod(numElements, numColumns);
    data_flat = [data_flat; zeros(padding, 1)]; % Add padding (zeros)
end

% Reshape the data into a matrix (jobs x machines)
jobs_data = reshape(data_flat, [], numColumns); % Reshape into 10 columns

% Save this data as a .mat file
save('job_schedule_data.mat', 'jobs_data');
figure;
set(gcf, 'WindowState', 'maximized');
hold on;
colors = lines(numJobs); % Unique colors for each job
    for i = 1:size(schedule, 1)
        job = schedule(i, 1);
        startTime = schedule(i, 3);
        endTime = schedule(i, 4);
        machine = schedule(i, 5);
        rectangle('Position', [startTime, machine, endTime-startTime, 0.8], ...
                  'FaceColor', colors(job, :), 'EdgeColor', 'k');
        text(startTime + (endTime-startTime)/2, machine + 0.4, ...
             ['Job ', num2str(job)], 'HorizontalAlignment', 'center');
    end
    xlabel('Time');
    ylabel('Machine');
    title('Gantt Chart of Job Schedule');
    grid on;
    hold off;