initial_statevector = [1;0;0;0]; %all qubits 0 

zero_state = [1;0];
one_state = [0;1];

X = [[0 1];[1 0]];
Y = [[];[]];
Z = [[];[]];

CNOT = [[1 0 0 0 ];[0 0 0 1];[0 0 1 0];[0 1 0 0]];

RX = @(theta) [[cos(theta/2) -1i*sin(theta/2)];[-1i*sin(theta/2) cos(theta/2)]];
CRX = @(theta) [[1 0 0 0];[0 cos(theta/2) 0 -1i*sin(theta/2)];[0 0 1 0];[0 -1i*sin(theta/2) 0 cos(theta/2)]];

RY = @(theta) [[cos(theta/2) -sin(theta/2)];[sin(theta/2) cos(theta/2)]];
CRY = @(theta) [[1 0 0 0];[0 cos(theta/2) 0 -1*sin(theta/2)];[0 0 1 0];[0 -1*sin(theta/2) 0 cos(theta/2)]];

RZ = @(lambda) [[exp(-1i*(lambda/2)) 0];[0 exp(1i*(lambda/2))]];
CRZ = @(lamba) [[1 0 0 0];[0 exp(-1i*(lambda/2)) 0 0];[0 0 1 0];[0 0 0 exp(1i*(lambda/2))]];

x = [-pi:0.01:pi];

z = [];

for n = 1:length(x)
    ztemp = RY(x(n)) * (RY(x(length(x)-n+1))*zero_state);
    z(n) = angle(ztemp(1) * 1i * ztemp(2));
end

