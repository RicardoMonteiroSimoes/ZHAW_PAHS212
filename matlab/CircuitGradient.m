

%% First circuit of |0> - RY(theta) - 
syms theta

initial_state = [1;0];

RY = [[cos(theta/2) -sin(theta/2)];
      [sin(theta/2) cos(theta/2)]];
  
dRY = diff(RY);

Q = transpose(eye(2)) * RY * eye(2);

A = transpose(initial_state) * RY * Q * dRY * initial_state;

gradient_function_a = A + ctranspose(A);

%% Second circuit of |0> - RY(alpha) - RY(beta) - 
syms alpha beta

RYa = [[cos(alpha/2) -sin(alpha/2)];
      [sin(alpha/2) cos(alpha/2)]];
  
RYb = [[cos(beta/2) -sin(beta/2)];
      [sin(beta/2) cos(beta/2)]];

G = RYb * RYa;
dG = diff(G);

Q = transpose(eye(2)) * G * eye(2);



A = transpose(initial_state) * G * Q * dG * initial_state;

gradient_function_b = A + ctranspose(A);
  
 
 