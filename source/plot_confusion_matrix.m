function plot_confusion_matrix(predict, ground_truth, name)
figure;
subplot(211);
N_class = 10;
C = confusionmat(predict,ground_truth);
labels = unique(ground_truth);
percent_c = 100*C./(repmat(sum(C,1), N_class, 1));
heatmap(labels, labels, percent_c);
xlabel('Predict');
ylabel('Ground truth');
title('Confusion matrix');

subplot(212);
[n1, p1] = hist(ground_truth); bar(p1,n1,'facecolor','r')
title(['Histogram of ',name,' data']);
end