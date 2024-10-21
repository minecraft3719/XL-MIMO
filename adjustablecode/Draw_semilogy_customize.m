clc
clear all
close all
loss_org = table2array(readtable('mr_Son_training_result_original_mixed_SNR\loss_history.txt'));
loss = table2array(readtable('mr_Son_training_result_single_SNR\loss_history.txt'));

figure(1)
plot(loss_org,'-o', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00", 'MarkerIndices', 1:5:length(loss_org))
hold on
plot(loss,'-o', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00", 'MarkerIndices', 1:5:length(loss))
legend("training loss overtime with multiple SNR","training loss overtime with single SNR",'Interpreter', 'latex', 'FontSize', 14, 'Edgecolor', 'white')
ylabel("nmse loss",'FontSize', 14, 'Interpreter','latex')
xlabel("epoch cycle",'FontSize', 14, 'Interpreter','latex')

hold off
all_marks = {'o','+','*','x','s','d','^','v','>','<','p','h'};

% path_directory = ('.');
% plot_file = dir([path_directory '/20241021-150319*.csv']);
% legend4plot = zeros(2,length(plot_file));
% Legend = cell(length(plot_file)+1,1);

figure(2)

nmse_result_plot_mixed_SNR = table2array(readtable('mr_Son_training_result_original_mixed_SNR\20241021-153637_nmseSummary_test.csv'));
semilogy(nmse_result_plot_mixed_SNR(:,6),nmse_result_plot_mixed_SNR(:,8),['-' all_marks{randi(length(all_marks))}], 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00");
Legend{1} = strcat("mixed SNR from -10 to 20 ","Lf = ",int2str(nmse_result_plot_mixed_SNR(1,4)),"; Ln = ",int2str(nmse_result_plot_mixed_SNR(1,5)));
hold on;
nmse_result_train_mixed_SNR = table2array(readtable('mr_Son_training_result_original_mixed_SNR\20241009-015033_nmseSummary_train.csv'));
semilogy(nmse_result_train_mixed_SNR(:,1),nmse_result_train_mixed_SNR(:,3),'-o', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00");
Legend{2} = "train nmse mixed SNR from -10 to 20";
grid minor;
hold on;
nmse_result_plot = table2array(readtable('mr_Son_training_result_single_SNR\20241021-150319_nmseSummary_test.csv'));
semilogy(nmse_result_plot(:,6),nmse_result_plot(:,8),['-' all_marks{randi(length(all_marks))}], 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00");
Legend{3} = strcat("single SNR = 10dB ","Lf = ",int2str(nmse_result_plot(1,4)),"; Ln = ",int2str(nmse_result_plot(1,5)));
hold on
nmse_result_train = table2array(readtable('mr_Son_training_result_single_SNR\20241021-170739_nmseSummary_train.csv'));
semilogy(nmse_result_train(:,1),nmse_result_train(:,3),'-o', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', "#FFFF00");
Legend{4} = "train nmse single SNR = 10dB";
grid minor;
xlabel('SNR','FontSize', 14, 'Interpreter','latex');
ylabel('nmse','FontSize', 14, 'Interpreter','latex');
legend(Legend,'Interpreter', 'latex', 'FontSize', 14, 'Edgecolor', 'white');
% set(gcf, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
% set(gcf, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
% saveas(gcf, 'NMSEvsSNR', 'pdf')
hold off