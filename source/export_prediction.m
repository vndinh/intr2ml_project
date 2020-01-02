function export_prediction(prediction, filename)
    f = fopen(filename,'w');
    N_pred = size(prediction, 1);
    fprintf(f,'id,class\n');
    for i = 1:N_pred
        fprintf(f,'%d,%d\n',i, prediction(i));
    end
    fclose(f);
end