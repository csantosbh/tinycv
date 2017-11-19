%close all;
clear;

model_param=2;

tdm=load(sprintf('%d_dmi.txt', model_param));
td2m=load(sprintf('%d_d2mi.txt', model_param));
x=tdm(:, 1);
x2=td2m(:, 1);
dmi=tdm(:, 2);
d2mi=td2m(:, 2);

% Plot axes
figure;
axes('xgrid', 'on', 'ygrid', 'on'); hold on;

plot(x, dmi, 'b');
plot(x2, d2mi, 'r');

dx = (x(3)-x(1));
x_dif = x(2:(end-1));
numerical_d2mi = (dmi(3:end)-dmi(1:(end-2)))./dx;
plot(x_dif, numerical_d2mi, 'g');

% Integrals
dxI = (x2(2)-x2(1));
plot(x2, dmi(1) + cumsum(d2mi * dxI), 'color', [ .8 .7  0.2]);
%{
dxI = (x_dif(2)-x_dif(1));
plot(x_dif, dmi(1) + cumtrapz(numerical_d2mi * dxI), 'color', [ .8 .7  0.2]);
%}

% Graph info
legend('Jacobian', 'Hessian', 'Num. Hessian', 'Hessian Integ.');

title('Derivative vs Hessian');

% Show difference between numerical and analytical hessians
%%{
figure;
plot(x_dif, d2mi(2:(end-1)) ./ numerical_d2mi, 'color', [255 219 15]/255);
%}
