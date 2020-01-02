function save_model(model, path, log_file, acc_train, acc_val)
t = datetime('now');
time = datestr(t,31);
time(time==' ')= '_';
time(time==':')= '_';
model_ckpt = [path,'model_ckpt_',time,'.mat'];
disp(['Save model: ', model_ckpt]);
save(model_ckpt, 'model');

f = fopen([path,log_file], 'a+');
fprintf(f, '%s, Training accuracy: %.2f, validation accuracy: %.2f\n', model_ckpt, acc_train, acc_val);
fclose(f);
end