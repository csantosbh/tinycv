close all;
clear;

tdm=load('x_dmi.txt');
td2m=load('x_d2mi.txt');
x=tdm(:, 1);
x2=td2m(:, 1);
dmi=tdm(:, 2);
d2mi=td2m(:, 2);

% Plot axes
figure;
axes('xgrid', 'on', 'ygrid', 'on'); hold on;

plot(x, dmi, 'b');
plot(x2, d2mi, 'r');

dx = 2 * (x(2)-x(1));
x_dif = x(2:(end-1));
numerical_d2mi = (dmi(3:end)-dmi(1:(end-2)))./dx;
plot(x_dif, numerical_d2mi, 'g');

% Integrals
dx2 = 2 * (x2(2)-x2(1));
plot(x2, dmi(1) + cumsum(d2mi * dx2), 'color', [ .8 .7  0.2]);

legend('Jacobian', 'Hessian', 'Num. Hessian', 'Hessian Integration');

% Show difference between numerical and analytical hessians
%{
figure;
plot(x_dif, d2mi(2:(end-1)) ./ numerical_d2mi, 'color', [255 219 15]/255);
%}
