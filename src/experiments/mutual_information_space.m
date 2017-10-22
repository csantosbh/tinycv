close all;

function data=plot_surf(filename, range)
  data = load(sprintf('data/%s', filename));
  tx = ty = linspace(-range, range, size(data, 1));

  figure;
  surfc(tx, ty, data, -10);

  title(filename);
  xlabel('x');
  ylabel('y');

  zmin = 0;
  zmax = max(data(:));
  axis([-range range -range range (zmin - 0.5) (zmax + 0.0)]);
endfunction


%{
%}
%plot_surf('perspective_jitter.txt', 20);
%plot_surf('perspective_bilinear.txt', 20);
%plot_surf('perspective_scaled_nn.txt', 20);
%plot_surf('perspective_scaled_bilinear.txt', 20);
%plot_surf('perspective_scaled_jitter.txt', 20);
%plot_surf('perspective_scaled_bilateral.txt', 20);

plot_surf('translate_scaled_bilinear.txt', 10);
plot_surf('translate_bilinear.txt', 10);

%{
ima=giw_load('image_a.oct');
imb=giw_load('image_b.oct');
%}

