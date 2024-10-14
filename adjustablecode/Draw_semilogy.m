clc
clear all
close all
loss = table2array(readtable('D:\XL-MIMO\adjustablecode\mr_Son_training_result\loss_history.txt'));

figure(1)
plot(loss)
legend("training loss overtime",FontSize=18)
ylabel("nmse loss",'FontSize',16)
xlabel("epoch cycle",'FontSize',16)



% SNR = graphing(:,6);
% estimation = graphing(:,7);
% prediction = graphing(:,8);
% figure
% semilogy(SNR,estimation)
% hold on 
% semilogy(SNR,prediction)
% 
% legend("estimation with noise")
% legend("using CNN to predict")

