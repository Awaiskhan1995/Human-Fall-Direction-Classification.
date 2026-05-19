% Define your cost function
% Example: Replace this with your actual cost function that evaluates the quality of the feature subset
fobj = @(fobj) Get_Functions_details(fobj, testFeatures_E, labels);

% MFO parameters
SearchAgents_no = 30; % Number of search agents
Function_name = 'F1'; % Name of the test function
Max_iteration = 1000; % Maximum number of iterations

% Load details of the selected benchmark function
[lb, ub, dim, ~] = Get_Functions_details(Function_name);

% Run MFO optimization
[Best_score, Best_pos, cg_curve] = MFO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj);

% Save selected features obtained from MFO
selectedFeatures = testFeatures_E(:, Best_pos);
save('selected_features.mat', 'selectedFeatures');

% Plot convergence curve
figure('Position', [284 214 660 290])
semilogy(cg_curve, 'Color', 'b')
title('Convergence curve')
xlabel('Iteration');
ylabel('Best flame (score) obtained so far');
axis tight
grid off
box on
legend('MFO')

display(['The best solution obtained by MFO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective function found by MFO is : ', num2str(Best_score)]);
