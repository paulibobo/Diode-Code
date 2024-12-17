
clear all
close all


% reservoir size
rsize = linspace(500,6000,12); 
tmse = [0.02807 0.02630 0.02209 0.02063 0.01995 0.01912 ...
        0.01859 0.01762 0.01779 0.01765 0.01735 0.01690];
ttime = [1.90 9.23 71.35 182.95 239.87 276.01 ...
        408.46 581.91 810.61 1332.85 1648.67 2491.32];

figure(20)

yyaxis right
plot(rsize,ttime,'-o','LineStyle','--','LineWidth',2,'Color','(0, 0.4470, 0.7410)');
hold on
yyaxis left
plot(rsize,tmse,'-o','LineStyle','--','LineWidth',2, 'Color','(0.8500, 0.3250, 0.0980)');
hold on

set(gca, 'FontSize', 14);
xlabel('Reservoir size');
lh=legend('Training MSE','Training time [s]');
lh.Position(1) = 0.5 - lh.Position(3)/2; 


