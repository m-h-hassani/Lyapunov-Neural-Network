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
B=0.005; % this is network training rate
%% neural network initialization
N=4; % number of hidden neuron
dim=4;
Wij_p_c =0.05*rand(N,dim);  %input to hidden weights
W1j_p_c =0.05*rand(1,N);   %hidden to output weights
W1j_p_c_new(:,:,2) = zeros(1,N);
Wij_p_c_new (:,:,2)= zeros(4,N);
Wij_p_e =0.05*rand(N,dim);  %input to hidden weights
W1j_p_e =0.05*rand(1,N);   %hidden to output weights
W1j_p_e_new(:,:,2) = zeros(1,N);
Wij_p_e_new(:,:,2) = zeros(4,N);
B_p_c = 0.005;
B_p_e = 0.005;
x_p_c  = x;
x_p_e = x;
%% neural network 
%initializing inputs 

y(1,1)=0;
y(2,1)=0;
y(3,1)=0;
e(1,1)=x(1)-y(1);
e(2,1)=x(2)-y(2);
e(3,1)=x(3)-y(3);
u_p_c(1,1)=0;
u_p_c(2,1)=0;
u_p_c(3,1)=0;
u_p_e(3,1) = 0.05;
% for i = 1:3
% %     u(i) = 0.005;
%     y(i) = 0.005;
%     e(i) = 0;
%     u_p_c (i) = 0.0012;
%    u_p_e (i)= 0;
% end

for k = 4:length(t)
  k
   y(k) = (0.8*y(k-1) + 2*u_p_c(k-1))/(1+1.5*y(k-2)*u_p_c(k-2));
   e(k)=x(k)-y(k);
%% NNP
err_p_c(k) = e(k);
%NNC*****************************************************************
   for j = 1:4
   for i= 1:4
       S_p_c(k,j) = tanh(Wij_p_c(j,i).*x_p_c(k-i+1));
   end
   end
   
% input layer
net1_p_c = [x_p_c(k);x_p_c(k-1);x_p_c(k-2);x_p_c(k-3)];
O1_p_c = net1_p_c;
% hidden layyer 

net2_p_c = Wij_p_c .*O1_p_c;
O2_p_c = [S_p_c(k,1),S_p_c(k,2),S_p_c(k,3),S_p_c(k,4)]; % not sure
%output layer
net3_p_c = W1j_p_c.*O2_p_c;
u_p_c(k) = net3_p_c(1,1)+ net3_p_c(1,2) + net3_p_c(1,3)+net3_p_c(1,4);


% updating weights %%%%%%%%%
% Updating W1j
d_p_c(k)= u_p_e(k-1); % must be changed ******************u_p_e(k-1)
sigma_p_c (k)  = sqrt(B_p_c)*err_p_c(k-1) + d_p_c(k);
for j = 1 :4
    W1j_p_c_new(k,j) = sigma_p_c (k) /(4*S_p_c(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_c_new(j,i) = 1/(4*x_p_c(k-i+1))*1/tanh(sigma_p_c(k)/(4*W1j_p_c(1,j))); % Not OK
    end
end
Wij_p_c = Wij_p_c_new(:,:,1);
W1j_p_c = W1j_p_c_new(4,:,1);

%NNE*****************************************************************

 for j = 1:4
   for i= 1:4
       S_p_e(k,j) = tanh(Wij_p_e(j,i).*x_p_e(k-i+1));
   end
 end
   
% input layer
net1_p_e = [x_p_e(k);x_p_e(k-1);x_p_e(k-2);x_p_e(k-3)];
O1_p_e = net1_p_e;
% hidden layyer 

net2_p_e = Wij_p_e.*O1_p_e;
O2_p_e = [S_p_e(k,1),S_p_e(k,2),S_p_e(k,3),S_p_e(k,4)]; % not sure
%output layer
net3_p_e = W1j_p_e.*O2_p_e;
u_p_e(k) = net3_p_e(1,1)+ net3_p_e(1,2) + net3_p_e(1,3)+net3_p_e(1,4);
err_p_e (k) = u_p_e(k) - u_p_c(k);
% updating weights 
% Updating W1j
d_p_e(k)=u_p_c(k); % must be changed
sigma_p_e (k)  = sqrt(B_p_e)*err_p_e(k-1) + d_p_e(k);
for j = 1 :4
    W1j_p_e_new(k,j) = sigma_p_e (k) /(4*S_p_e(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_e_new(j,i) = 1/(4*x_p_e(k-i+1))*1/tanh(sigma_p_e(k)/(4*W1j_p_e(1,j))); % Not OK
    end

end
Wij_p_e = Wij_p_e_new(:,:,1);
W1j_p_e = W1j_p_e_new(4,:,1);


% u(k)= u_p_c(k) ;

dbstop if naninf  % for debugg

end % end of sample k 
plot(t,y,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('u(k)')
title('system output')
legend('y(k)')

