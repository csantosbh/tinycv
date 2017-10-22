close all;

function data=plot_surf(filename, range)
  data = load(filename);
  tx = ty = linspace(-range, range, size(data, 1));

  figure;
  surf(tx, ty, data);

  title(filename);
  xlabel('x');
  ylabel('y');
endfunction


%{
%}
%plot_surf('perspective_jitter.txt', 20);
%plot_surf('perspective_bilinear.txt', 20);
%plot_surf('perspective_scaled_nn.txt', 20);
%plot_surf('perspective_scaled_bilinear.txt', 20);
%plot_surf('perspective_scaled_jitter.txt', 20);
%plot_surf('perspective_scaled_bilateral.txt', 20);

plot_surf('translate_scaled_bilinear.txt', 20);
plot_surf('translate_bilinear.txt', 20);

%{
ima=giw_load('image_a.oct');
imb=giw_load('image_b.oct');
%}

