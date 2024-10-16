clc
clear all
close all
loss = table2array(readtable('D:\XL-MIMO\adjustablecode\mr_Son_training_result\loss_history.txt'));

figure(1)
plot(loss)
legend("training loss overtime",FontSize=18)
ylabel("nmse loss",'FontSize',16)
xlabel("epoch cycle",'FontSize',16)

path_directory = ('D:\XL-MIMO\adjustablecode');
plot_file = dir([path_directory '/20241015*.csv']);

figure(2)
legend4plot = zeros(2,length(plot_file));
Legend = cell(length(plot_file),1);
for i = 1:length(plot_file)
    nmse_result = plot_file(i).name;
    nmse_result_plot = table2array(readtable(nmse_result));
    semilogy(nmse_result_plot(:,6),nmse_result_plot(:,8));
    Legend{i} = strcat("Lf = ",int2str(nmse_result_plot(1,4)),"; Ln = ",int2str(nmse_result_plot(1,5)));
    hold on
end
grid on
xlabel('SNR')
ylabel('nmse')
legend(Legend);
hold off