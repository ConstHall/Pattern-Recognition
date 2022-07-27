x = randn(2000,2) * [2 1;1 2];
mean_x = mean(x); % 均值
cov_x = cov(x); % 协方差矩阵
[ev,ed] = eigs(cov_x); % 特征向量
[row,col] = size(x);
u = repmat(mean_x , row, 1);
x1 = (x - u) * ev * inv(sqrt(ed));
hold on;
scatter(x(:,1), x(:,2));
scatter(x1(:, 1), x1(:, 2));
