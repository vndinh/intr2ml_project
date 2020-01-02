function [X,Y] = smote(x_train,y_train,k,threshold)
  %% Arguments %%
  % x_train: training data point
  % y_train: training data label
  % k      : number of nearest neighbors
  
  X = x_train;
  Y = y_train;
  [m,~] = size(X);
  
  H = zeros(10,1);
  for i = 1:10
    H(i) = sum(y_train==i);
  end
  
  for i = 1:10
    if H(i) < (threshold*m)
      Yidx = find(y_train==i);
      Xm = X(Yidx,:);
      K = knn(Xm,Xm,k);
      for j = 1:H(i)
        idx = randi(k);
        R = Xm(j,:) + (Xm(j,:)-Xm(K(idx,j),:))*rand(1);
        n = size(X,1);
        row = randi(n);
        %X = insert_row(X,R,row);
        %Y = insert_row(Y,i,row);
        X = [X; R];
        Y = [Y; i];
        %X = [X(1:row,:); R; X((row+1):n,:)];
        %Y = [Y(1:row,:); i; Y((row+1):n,:)];
      end
    end
  end
end

function K = knn(x_input,x_train,k)
  %% Arguments %%
  % x_input: inputs to classify
  % x_train: training data
  % k: number of nearest neighbor
  % class: predicted class

  m = size(x_train,1);    % Number of the training data points
  p = size(x_input,1);    % Number of the test data points

  % Calculating distance by norm 2
  D = zeros(m,p);
  for i = 1:p
    for j = 1:m
      d = x_input(i,:) - x_train(j,:);
      D(j,i) = d * d';
    end
  end

  % Finding the index of k nearest points in the training set
  [~,Didx] = sort(D,1);
  K = Didx(2:k+1,:);
end

function A = insert_row(X,R,k)
  [m,~] = size(X);
  A = [X(1:k,:); R; X(k+1:m,:)];
end

