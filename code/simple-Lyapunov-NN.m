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
W1j_new(:,:,2) = zeros(1,N);
Wij_new (:,:,2)= zeros(4,N);

%% neural network 
%initializing inputs 
y(1,1)=0;
y(2,1)=0;
y(3,1)=0;
e(1,1)=x(1)-y(1);
e(2,1)=x(2)-y(2);
e(3,1)=x(3)-y(3);
u(1,1)=0;
u(2,1)=0;
u(3,1)=0;
%u_p_e(3,1) = 0.05;

for k = 4:length(t)
  k
   y(k) = (0.8*y(k-1) + 2*u(k-1))/(1+1.5*y(k-2)*u(k-2)); %plant
   e(k)=x(k)-y(k);

err(k) = e(k);
   for j = 1:4
   for i= 1:4
       S(k,j) = tanh(Wij(j,i).*x(k-i+1));
   end
   end
   
% input layer
net1 = [x(k);x(k-1);x(k-2);x(k-3)];
O1= net1;
% hidden layyer 

net2 = Wij .*O1;
O2 = [S(k,1),S(k,2),S(k,3),S(k,4)]; % not sure
%output layer
net3 = W1j.*O2;
u(k) = net3(1,1)+ net3(1,2) + net3(1,3)+net3(1,4);


% updating weights %%%%%%%%%
% Updating W1j
d(k)= 0.05; % must be changed ******************u_p_e(k-1)
sigma_(k)  = sqrt(B)*err(k-1) + d(k);
for j = 1 :4
    W1j_new(k,j) = sigma (k) /(4*S(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_new(j,i) = 1/(4*x(k-i+1))*1/tanh(sigma(k)/(4*W1j(1,j))); % Not OK
    end
end
Wij = Wij_new(:,:,1);
W1j = W1j_new(4,:,1);

end % end of sample k 
plot(t,y,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('u(k)')
title('system output')
legend('y(k)')
