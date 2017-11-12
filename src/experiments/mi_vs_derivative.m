%close all;
clear;

mi=load('x_mi.txt');
tdm=load('x_dmi.txt');
x=tdm(:, 1);
dmi=tdm(:, 2);

% Plot axes
figure;
axes('xgrid', 'on', 'ygrid', 'on'); hold on;
X=[min(x) 0; max(x) 0];
Y=1.1*[0 min(min(mi, dmi)(:)); 0 max(max(mi, dmi)(:))];
%axis([X(1, 1) X(2, 1) Y(1, 2) Y(2, 2)]);
%plot(X(:, 1), X(:, 2), 'color', [0 0 0]);
%plot(Y(:, 1), Y(:, 2), 'color', [0 0 0]);

plot(x, mi, 'b');
plot(x, dmi, 'r');

dx = 2 * (x(2)-x(1));
x_dif = x(2:(end-1));
numerical_dmi = (mi(3:end)-mi(1:(end-2)))./dx;
plot(x_dif, numerical_dmi, 'g');

% Integrals
plot(x, mi(1) + cumtrapz(dmi * dx * 0.5), 'color', [ .8 .7  0.2]);
%plot(x_dif, mi(1) + cumtrapz(numerical_dmi * dx * 0.5), 'color', [ .8 .7  0.2]);

% Numeric derivative of gradient
figure;
numerical_d2mi = (dmi(3:end)-dmi(1:(end-2)))./dx;
plot(x_dif, numerical_d2mi, 'color', [ .8 .7  0.2]);

% Show difference between numerical and analytical gradients
%figure;
%plot(x_dif, dmi(2:(end-1)) ./ numerical_dmi', 'color', [255 219 15]/255);
