clc
clear all
close all
%% lyapanov based neural network
% Input signal
h=0.002;
t=0:h:3.5;
t=t';
for k=1:length(t)
    x(k,1)=1;
    
end
B=1; % this is network training rate
%% neural network initialization
N=4; % number of hidden neuron
dim=4;
Wij=0.05*rand(N,dim);  %input to hidden weights
W1j =0.05*rand(1,N);   %hidden to output weights

%% neural network 
%initializing inputs 
x(1)=0.5;
x(2)=0.5;
x(3)=0.5;
sum = 0;
err(3) = 0;
for k = 4:length(t)
   for j = 1:4
   % S(k,j) = tanh((Wij(1,j)*x(k-1+1) + Wij(2,j)*x(k-2+1)+Wij(3,j)*x(k-3+1)+Wij(4,j)*x(k-4+1))); % check again
   for i= 1:4
       S(k,j) = tanh(Wij(j,i).*x(k-i+1));
   end
   end
   
% input layer
net1 = [x(k);x(k-1);x(k-2);x(k-3)];
O1 = net1;
% hidden layyer 

net2 = Wij*O1;
O2 = [S(k,1),S(k,2),S(k,3),S(k,4)]; % not sure
%output layer
net3 = W1j.*O2;
%O3 = W1j(1,1)*S(k,1)+W1j(1,2)*S(k,2)+W1j(1,3)*S(k,3)+W1j(1,4)*S(k,4); %rewrite
% O3 = [W1j(k,1).*S(k,1);W1j(k,2).*S(k,2);W1j(k,3).*S(k,3);W1j(k,4).*S(k,4)]
u(k) = net3(1,1)+ net3(1,2) + net3(1,3)+net3(1,4);
err(k) = u(k) -x(k);

%% updating weights 
% Updating W1j
d(k)=0; % must be changed
sigma (k)  = sqrt(B)*err(k-1) + d(k);
for j = 1 :4
    W1j_new(k,j) = sigma (k) /(4*S(k-1,j));
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_new(j,i) = 1/(4*x(k-i+1))*1/tanh(sigma(k)/(4*W1j(1,j))); % Not OK
    end

end
Wij = Wij_new;
W1j = W1j_new;

end % end of sample k 
plot(t,u,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('u(k)')
title('system output with NN-PID')
legend('y(k)')
