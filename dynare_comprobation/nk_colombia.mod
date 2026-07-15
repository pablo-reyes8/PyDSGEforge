// Minimal Dynare/Octave reference model used by the parity check.
// Run from this directory: dynare nk_colombia.mod

var x pi i;
varexo eps_d eps_s eps_m;
parameters beta sigma kappa phi_pi phi_x;

beta   = 0.99;
sigma  = 1.50;
kappa  = 0.10;
phi_pi = 1.50;
phi_x  = 0.50;

model(linear);
  x  = x(+1) - (1/sigma)*(i - pi(+1)) + eps_d;
  pi = beta*pi(+1) + kappa*x + eps_s;
  i  = phi_pi*pi + phi_x*x + eps_m;
end;

initval;
  x = 0;
  pi = 0;
  i = 0;
end;

steady(nocheck);
check;
varobs x pi i;

estimated_params;
  beta,   beta_pdf,      0.99, 0.005;
  sigma,  gamma_pdf,     1.50, 0.50;
  kappa,  gamma_pdf,     0.10, 0.05;
  phi_pi, gamma_pdf,     1.50, 0.30;
  phi_x,  gamma_pdf,     0.50, 0.20;
  stderr eps_d, inv_gamma_pdf, 0.20, 0.10;
  stderr eps_s, inv_gamma_pdf, 0.20, 0.10;
  stderr eps_m, inv_gamma_pdf, 0.20, 0.10;
end;

estimated_params_init;
  beta,   0.99;
  sigma,  1.50;
  kappa,  0.10;
  phi_pi, 1.50;
  phi_x,  0.50;
end;

estimation(
  datafile='data/colombia_nk_quarterly.csv',
  first_obs=1,
  prefilter=1,
  mode_compute=6,
  mh_replic=1000,
  mh_drop=0.5,
  mh_nblocks=1,
  irf=20
);
