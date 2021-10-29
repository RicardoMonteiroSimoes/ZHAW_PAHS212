syms alpha_0 alpha_1 alpha_2 alpha_3 omega_0 omega_10 omega_1 omega_11 omega_2 omega_12 omega_3 omega_13

state_zero = [1;0];

RY1 = [[cos(alpha_0/2) -sin(alpha_0/2)];[sin(alpha_0/2) cos(alpha_0/2)]];
RY2 = [[cos(alpha_1/2) -sin(alpha_1/2)];[sin(alpha_1/2) cos(alpha_1/2)]];
RY3 = [[cos(alpha_2/2) -sin(alpha_2/2)];[sin(alpha_2/2) cos(alpha_2/2)]];
RY4 = [[cos(alpha_3/2) -sin(alpha_3/2)];[sin(alpha_3/2) cos(alpha_3/2)]];

RYw = [[cos(omega_0/2) -sin(omega_0/2)];[sin(omega_0/2) cos(omega_0/2)]];

CRY10 = [[1 0 0 0];[0 1 0 0];[0 0 cos(omega_10/2) -sin(omega_10/2)];[0 0 sin(omega_10/2) cos(omega_10/2)]];
CRY11 = kron(eye(2), [[1 0 0 0];[0 1 0 0];[0 0 cos(omega_11/2) -sin(omega_11/2)];[0 0 sin(omega_11/2) cos(omega_11/2)]]);
CRY12 = kron(eye(2), [[1 0 0 0];[0 1 0 0];[0 0 cos(omega_12/2) -sin(omega_12/2)];[0 0 sin(omega_12/2) cos(omega_12/2)]]);
CRY13 = kron(eye(4), [[1 0 0 0];[0 cos(omega_13/2) 0 -sin(omega_13/2)];[0 0 1 0];[0 sin(omega_12/2) 0 cos(omega_12/2)]]);

s1 = RY1*state_zero;
s2 = RYw*s1;
qubittwo = RY2*state_zero;
qubitthree = RY3 * state_zero;
qubitfour = RY4 * state_zero;
s3 = kron(s2, qubittwo);
s4 = CRY10 * s3;

RYw_1 = [[cos(omega_1/2) -sin(omega_1/2)];[sin(omega_1/2) cos(omega_1/2)]];
RYw_1_wide = kron(eye(2), RYw_1);
s5 = RYw_1_wide * s4;
threeqbits = kron(s5, qubitthree);
s6 = CRY11 * threeqbits;
RYw_2 = [[cos(omega_2/2) -sin(omega_2/2)];[sin(omega_2/2) cos(omega_2/2)]];
RYw_2_wide = kron(eye(4), RYw_2);
s7 = RYw_2_wide * s6;
s8 = kron(s7, qubitfour);
CRY12_wide = kron(eye(2), CRY12);
s9 = CRY12_wide * s8;
RYw_3 = [[cos(omega_3/2) -sin(omega_3/2)];[sin(omega_3/2) cos(omega_3/2)]];
RYw_3_wide = kron(eye(8), RYw_3);
s10 = RYw_3_wide * s9;
s11 = CRY13 * s10;

alpha = [pi pi/2 3*pi/4 pi/4];
alpha = alpha ./ sum(alpha);
omega = [pi/10 pi/5 pi pi/2 pi/15 pi/2 pi/4 pi/6];
omega = omega ./ sum(omega);

psi9 = s11;
psi9 = subs(psi9, alpha_0, alpha(1));
psi9 = subs(psi9, alpha_1, alpha(2));
psi9 = subs(psi9, alpha_2, alpha(3));
psi9 = subs(psi9, alpha_3, alpha(4));
psi9 = subs(psi9, omega_0, omega(1));
psi9 = subs(psi9, omega_1, omega(2));
psi9 = subs(psi9, omega_2, omega(3));
psi9 = subs(psi9, omega_3, omega(4));
psi9 = subs(psi9, omega_10, omega(5));
psi9 = subs(psi9, omega_11, omega(6));
psi9 = subs(psi9, omega_12, omega(7));
psi9 = subs(psi9, omega_13, omega(8));

psi10 = eval(psi9);
fprintf('Sum should be 1.0: %.9f\n', sum(psi10.^2));