function live_plot(training)
    first_time = training.index_graph == 1;
    if first_time
        %% Plot validation and training error
        subplot(211);
        plot(training.index_graph, training.graph_val(training.index_graph), 'ro-', 'MarkerSize',3);
        plot(training.index_graph, training.graph_train(training.index_graph), 'bo-', 'MarkerSize',3);
        title_text = ['Performance - epoch: ', num2str(training.N_EPOCH), ' / Validation per epoch: ', num2str(training.val_per_epoch)];
        title(title_text);
        drawnow;
        axis([0 training.N_EPOCH 0 1]);
        hold on;
        
        subplot(212);
        plot(training.index_graph, training.loss(training.index_graph), 'ro-', 'MarkerSize',3)
        title_text = ['Loss - epoch: ', num2str(training.N_EPOCH), ' / Validation per epoch: ', num2str(training.val_per_epoch)];
        title(title_text);
        drawnow;
        axis([0 training.N_EPOCH 0 5]);
        hold on;
    else
        subplot(211);
        plot(training.index_graph, training.graph_val(training.index_graph), 'ro-', 'MarkerSize',3); hold on;
        plot(training.index_graph, training.graph_train(training.index_graph), 'bo-', 'MarkerSize',3);
        line([training.index_graph-1, training.index_graph], [training.graph_val(training.index_graph-1), training.graph_val(training.index_graph)], 'Color','red');
        line([training.index_graph-1, training.index_graph], [training.graph_train(training.index_graph-1), training.graph_train(training.index_graph)], 'Color','blue');
        drawnow;
        axis([0 training.index_graph+50 0 1]);
        
        subplot(212);
        plot(training.index_graph, training.loss(training.index_graph), 'ro-', 'MarkerSize',3);
        line([training.index_graph-1, training.index_graph], [training.loss(training.index_graph-1), training.loss(training.index_graph)], 'Color','red');
        drawnow;
        axis([0 training.index_graph+50 0 5]);
    end
end