syms alpha beta gamma delta epsilon

zero_state = [1;0]

RX = @(theta) [[cos(theta/2) -1i*sin(theta/2)];[-1i*sin(theta/2) cos(theta/2)]];
CRX = @(theta) [[1 0 0 0];[0 cos(theta/2) 0 -1i*sin(theta/2)];[0 0 1 0];[0 -1i*sin(theta/2) 0 cos(theta/2)]];

RY = @(theta) [[cos(theta/2) -sin(theta/2)];[sin(theta/2) cos(theta/2)]];
CRY = @(theta) [[1 0 0 0];[0 cos(theta/2) 0 -1*sin(theta/2)];[0 0 1 0];[0 -1*sin(theta/2) 0 cos(theta/2)]];

RZ = @(lambda) [[exp(-1i*(lambda/2)) 0];[0 exp(1i*(lambda/2))]];
CRZ = @(lamba) [[1 0 0 0];[0 exp(-1i*(lambda/2)) 0 0];[0 0 1 0];[0 0 0 exp(1i*(lambda/2))]];

%circuits have to be reverse order!
circuit_XOR = RY(delta) * (RY(gamma) * (RY(beta) * (RY(alpha) * zero_state)));