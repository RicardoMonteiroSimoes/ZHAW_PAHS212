syms alpha_0 alpha_1 alpha_2 alpha_3 omega_0 omega_10 omega_1

state_zero = [1;0];

RY1 = [[cos(a_0/2) -sin(a_0/2)];[sin(a_0/2) cos(a_0/2)]];
RY2 = [[cos(a_1/2) -sin(a_1/2)];[sin(a_1/2) cos(a_1/2)]];

RYw = [[cos(w_0/2) -sin(w_0/2)];[sin(w_0/2) cos(w_0/2)]];

CRY10 = [[1 0 0 0];[0 1 0 0];[0 0 cos(w_10/2) -sin(w_10/2)];[0 0 sin(w_10/2) cos(w_10/2)]];

psi1 = RY1*state_zero;
psi2 = RYw*psi1;

qubittwo = RY2*state_zero;
psi3 = kron(psi2, qubittwo);
psi4 = CRY10 * psi3;

RYw_1 = [[cos(w_1/2) -sin(w_1/2)];[sin(w_1/2) cos(w_1/2)]];
RYw_1_wide = kron(eye(2), RYw_1);
psi5 = RYw_1_wide * psi4;