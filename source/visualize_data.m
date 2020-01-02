
function visualize_data(x,y, mode)
addpath('./visualization/');
K_NEIGHBORS = 100;
D = 3;

N_class = size(unique(y),1);

% Projecting data
if strcmp(mode, 'lle')
    z = lle(x', K_NEIGHBORS, D);
    z = z';
elseif strcmp(mode, 'pca')
    [z, ~, ~] = train_PCA(x, D);
end

C = [1 0 0; %1 - red
     0 1 0; %2 - green
     0 0 1; %3 - blue
     1 1 0; %4 - yellow
     1 0 1; %5 - magenta
     0 1 1; %6 - cyan
     0 0 0; %7
     1 0 0; %8 
     0 0.4 0.4; %9
     0.6 0.4 0]; %10
for i = 1:N_class
    scatter3(z(y==i,1),z(y==i,2), z(y==i,3),[],C(i,:),'filled');
    hold on;
end
legend('class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10')
end