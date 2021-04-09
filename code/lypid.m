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
Wij_d_c =0.05*rand(N,dim);  %input to hidden weights
W1j_d_c =0.05*rand(1,N);   %hidden to output weights
W1j_d_c_new(:,:,2) = zeros(1,N);
Wij_d_c_new(:,:,2) = zeros(4,N);
Wij_d_e =0.05*rand(N,dim);  %input to hidden weights
W1j_d_e =0.05*rand(1,N);   %hidden to output weights
W1j_d_e_new(:,:,2) = zeros(1,N);
Wij_d_e_new(:,:,2) = zeros(4,N);
Wij_p_c =0.05*rand(N,dim);  %input to hidden weights
W1j_p_c =0.05*rand(1,N);   %hidden to output weights
W1j_p_c_new(:,:,2) = zeros(1,N);
Wij_p_c_new(:,:,2) = zeros(4,N);
Wij_p_e =0.05*rand(N,dim);  %input to hidden weights
W1j_p_e =0.05*rand(1,N);   %hidden to output weights
W1j_p_e_new(:,:,2) = zeros(1,N);
Wij_p_e_new(:,:,2) = zeros(4,N);
Wij_i_c =0.05*rand(N,dim);  %input to hidden weights
W1j_i_c =0.05*rand(1,N);   %hidden to output weights
W1j_i_c_new(:,:,2) = zeros(1,N);
Wij_i_c_new(:,:,2) = zeros(4,N);
Wij_i_e =0.05*rand(N,dim);  %input to hidden weights
W1j_i_e =0.05*rand(1,N);   %hidden to output weights
W1j_i_e_new(:,:,2) = zeros(1,N);
Wij_i_e_new(:,:,2) = zeros(4,N);
B_p_c = 0.25;
B_p_e = 0.75;
B_i_c = 0.00005;
B_i_e = 0.8;
B_d_c = 0.025;
B_d_e = 0.8;
x_p_c  = x;
x_p_e = x;
x_i_c  = x;
x_i_e = x;
x_d_c  = x;
x_d_e = x;
%% neural network 
%initializing inputs 
% x(1)=0.5;
% x(2)=0.5;
% x(3)=0.5;
% u(3) = 0;
% u(2) = 0;
% y(1)= 0;
% y(2) = 0;
% y(3) = 0;
% u_p_c (3) = 0;
% u_i_c (3) = 0;
% for i = 1:3
%     x(i) = 0.5;
%     u(i) = 0;
%     y(i) = 0;
%     e(i) = 0;
%     u_p_c (i) = 0;
%     u_i_c (i) = 0;
%     u_d_c (i) = 0;
% end
% sum = 0;
% err_d_c(3) = 0;
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
u_i_c(1,1)=0;
u_i_c(2,1)=0;
u_i_c(3,1)=0;
u_i_e(3,1) = 0.05;
u_d_c(1,1)=0;
u_d_c(2,1)=0;
u_d_c(3,1)=0;
u_d_e(3,1) = 0.05;
u(3,1) = 0;
u(2,1) = 0;
Kp=0.3;
Ki=0.2;
Kd=0.01;
for k = 4:length(t)
  k
    y(k) = (0.8*y(k-1) + 2*u(k-1))/(1+1.5*y(k-2)*u(k-2));
%    u(k)= u_p_c(k) + u_i_c(k) + u_d_c(k);
   e(k)=x(k)-y(k);
   
%    %% disturbance
%     if t(k-1)==1
%         u(k)=10*u(k); %first dis
%     elseif t(k)==1.8
%         u(k)=u(k)+0.89; %second dis
%     end
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
O2_p_c = [S_p_c(k,1),S_p_c(k,2),S_p_c(k,3),S_p_c(k,4)]; 
%output layer
net3_p_c = W1j_p_c.*O2_p_c;
u_p_c(k) = net3_p_c(1,1)+ net3_p_c(1,2) + net3_p_c(1,3)+net3_p_c(1,4);


% updating weights %%%%%%%%%
% Updating W1j
d_p_c(k)= u_p_e(k-1); 
sigma_p_c (k)  = sqrt(B_p_c)*err_p_c(k-1) + d_p_c(k);
for j = 1 :4
    W1j_p_c_new(k,j) = sigma_p_c (k) /(4*S_p_c(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_c_new(j,i) = 1/(4*x_p_c(k-i+1))*atanh(sigma_p_c(k)/(4*W1j_p_c(1,j))); % Not OK
    end
end
Wij_p_c = Wij_p_c_new(:,:,1);
W1j_p_c = W1j_p_c_new(4,:,1); %first 3 rows are zero

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
O2_p_e = [S_p_e(k,1),S_p_e(k,2),S_p_e(k,3),S_p_e(k,4)]; 
%output layer
net3_p_e = W1j_p_e.*O2_p_e;
u_p_e(k) = net3_p_e(1,1)+ net3_p_e(1,2) + net3_p_e(1,3)+net3_p_e(1,4);

err_p_e (k) = u_p_e(k) - u_p_c(k);

% updating weights 
% Updating W1j
d_p_e(k)=u_p_c(k); 
sigma_p_e (k)  = sqrt(B_p_e)*err_p_e(k-1) + d_p_e(k);
for j = 1 :4
    W1j_p_e_new(k,j) = sigma_p_e (k) /(4*S_p_e(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_e_new(j,i) = 1/(4*x_p_e(k-i+1))*atanh(sigma_p_e(k)/(4*W1j_p_e(1,j))); % Not OK
    end

end
Wij_p_e = Wij_p_e_new(:,:,1);
W1j_p_e = W1j_p_e_new(4,:,1); % only the 4th row has positive num


%% NNI
%NNC *****************************************************************
err_i_c  =  0;     %int(e(k));
for i = 1:k
    err_i_c = err_i_c + e(i);
end
 for j = 1:4
   for i= 1:4
       S_i_c(k,j) = tanh(Wij_i_c(j,i).*x_i_c(k-i+1));
   end
   end
   
   
% input layer
net1_i_c = [x_i_c(k);x_i_c(k-1);x_i_c(k-2);x_i_c(k-3)];
O1_i_c = net1_i_c;
% hidden layyer 

net2_p_c = Wij_p_c .*O1_p_c;
O2_i_c = [S_i_c(k,1),S_i_c(k,2),S_i_c(k,3),S_i_c(k,4)]; % not sure
%output layer
net3_i_c = W1j_i_c.*O2_i_c;
u_i_c(k) = net3_i_c(1,1)+ net3_i_c(1,2) + net3_i_c(1,3)+net3_i_c(1,4);

err_i_c(k) = u_i_c(k) -x(k); %?????????????

% updating weights 
% Updating W1j
d_i_c(k)= u_i_e(k-1); % must be changed ******************u_p_e(k-1)
sigma_i_c (k)  = sqrt(B_i_c)*err_i_c(k-1) + d_i_c(k);
for j = 1 :4
    W1j_i_c_new(k,j) = sigma_i_c (k) /(4*S_i_c(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_i_c_new(j,i) = 1/(4*x_i_c(k-i+1))*atanh(sigma_i_c(k)/(4*W1j_i_c(1,j))); % Not OK
    end
end
Wij_i_c = Wij_i_c_new(:,:,1);
W1j_i_c = W1j_i_c_new(4,:,1);



%NNE*****************************************************************
x_i_e (k) = x(k);
% err_i_e (k) = u_i_e(k) - u_i_c(k);
for j = 1:4
   for i= 1:4
       S_i_e(k,j) = tanh(Wij_i_e(j,i).*x_i_e(k-i+1));
   end
 end
   
% input layer
net1_i_e = [x_i_e(k);x_i_e(k-1);x_i_e(k-2);x_i_e(k-3)];
O1_i_e = net1_i_e;
% hidden layyer 

net2_i_e = Wij_i_e.*O1_i_e;
O2_i_e = [S_i_e(k,1),S_i_e(k,2),S_i_e(k,3),S_i_e(k,4)]; % not sure
%output layer
net3_i_e = W1j_i_e.*O2_i_e;
u_i_e(k) = net3_i_e(1,1)+ net3_i_e(1,2) + net3_i_e(1,3)+net3_i_e(1,4);
err_i_e (k) = u_i_e(k) - u_i_c(k);
% updating weights 
% Updating W1j
d_i_e(k)=u_i_c(k); % must be changed
sigma_i_e (k)  = sqrt(B_i_e)*err_i_e(k-1) + d_i_e(k);
for j = 1 :4
    W1j_i_e_new(k,j) = sigma_i_e (k) /(4*S_i_e(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_i_e_new(j,i) = 1/(4*x_i_e(k-i+1))*atanh(sigma_i_e(k)/(4*W1j_i_e(1,j))); % Not OK
    end

end
Wij_i_e = Wij_i_e_new(:,:,1);
W1j_i_e = W1j_i_e_new(4,:,1);



%% NND
%NNC*****************************************************************
% x_d_c (k) = x(k);
err_d_c (k) = e(k) - e(k-1);
 for j = 1:4
   for i= 1:4
       S_d_c(k,j) = tanh(Wij_d_c(j,i).*x_d_c(k-i+1));
   end
   end
   
% input layer
net1_d_c = [x_d_c(k);x_d_c(k-1);x_d_c(k-2);x_d_c(k-3)];
O1_d_c = net1_d_c;
% hidden layyer 

net2_d_c = Wij_d_c .*O1_d_c;
O2_d_c = [S_d_c(k,1),S_d_c(k,2),S_d_c(k,3),S_d_c(k,4)]; % not sure
%output layer
net3_d_c = W1j_d_c.*O2_d_c;
u_d_c(k) = net3_d_c(1,1)+ net3_d_c(1,2) + net3_d_c(1,3)+net3_d_c(1,4);


% updating weights %%%%%%%%%
% Updating W1j
d_d_c(k)= u_d_e(k-1); % must be changed ******************u_p_e(k-1)
sigma_d_c (k)  = sqrt(B_d_c)*err_d_c(k-1) + d_d_c(k);
for j = 1 :4
    W1j_d_c_new(k,j) = sigma_d_c (k) /(4*S_d_c(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_d_c_new(j,i) = 1/(4*x_d_c(k-i+1))*atanh(sigma_d_c(k)/(4*W1j_d_c(1,j))); % Not OK
    end
end
Wij_d_c = Wij_d_c_new(:,:,1);
W1j_d_c = W1j_d_c_new(4,:,1);

%NNE*****************************************************************
x_d_e (k) = x(k);
% err_d_e (k) = u_d_e(k) - u_d_c(k);
  for j = 1:4
   for i= 1:4
       S_d_e(k,j) = tanh(Wij_d_e(j,i).*x_d_e(k-i+1));
   end
 end
   
% input layer
net1_d_e = [x_d_e(k);x_d_e(k-1);x_d_e(k-2);x_d_e(k-3)];
O1_d_e = net1_d_e;
% hidden layyer 

net2_d_e = Wij_d_e.*O1_d_e;
O2_d_e = [S_d_e(k,1),S_d_e(k,2),S_d_e(k,3),S_d_e(k,4)]; % not sure
%output layer
net3_d_e = W1j_d_e.*O2_d_e;
u_d_e(k) = net3_d_e(1,1)+ net3_d_e(1,2) + net3_d_e(1,3)+net3_d_e(1,4);
err_d_e (k) = u_d_e(k) - u_d_c(k);
% updating weights 
% Updating W1j
d_d_e(k)=u_d_c(k); % must be changed
sigma_d_e (k)  = sqrt(B_d_e)*err_d_e(k-1) + d_d_e(k);
for j = 1 :4
    W1j_d_e_new(k,j) = sigma_d_e (k) /(4*S_d_e(k,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_d_e_new(j,i) = 1/(4*x_d_e(k-i+1))*atanh(sigma_d_e(k)/(4*W1j_d_e(1,j))); % Not OK
    end

end
Wij_d_e = Wij_d_e_new(:,:,1);
W1j_d_e = W1j_d_e_new(4,:,1);

Kp = u_p_c(k)/3.33;
Ki = u_i_c(k)/3.33;
Kd = u_d_c(k)/20;
 K(k,1) = Kp;
 K(k,2) = Ki;
 K(k,3) = Kd;
% u(k)= u(k-1) + u_p_c(k)*(e(k)-e(k-1)) + u_i_c(k)*e(k) + u_d_c(k)*(e(k)-2*e(k-1)+e(k-2));
u(k)= u(k-1) + Kp*(e(k)-e(k-1)) + Ki*e(k) + Kd*(e(k)-2*e(k-1)+e(k-2));

j(k) = 0.5*e(k)^2;

end % end of sample k 
plot(t,y,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('Y(k)')
title('system output')

figure
plot(t,j,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('J(k)')
title('J(k)')

figure
plot(t,u,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('u(k)')
title('control signal')
legend('u(k)')

figure
plot(t,e,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('e(k)')
title('system error')