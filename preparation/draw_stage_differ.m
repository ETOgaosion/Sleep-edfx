x = linspace(-12.0,12.0);

% tanh = 2./(1.0 + exp(-2.0 * x)) - 1;
% plot(x,tanh,'LineWidth',3);
sigma=1./(1.0+exp(-1.0*x));
plot(x,sigma,'LineWidth',3);
axis([-10.5 10.5 -1.2 1.2]);
% set(gca, 'XGrid','on');  % X轴的网格
% set(gca, 'YGrid','on');  % Y轴的网格;

set(gca,'FontSize',18);
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

box off;