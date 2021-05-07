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
    r(k,1)=1;
end
%% ANN neural network initialization
N_ANN=4; % number of hidden neuron
dim_ANN=4;
eta_ANN=0.01; % learning rate
alf=0.04; % momentom
Wij_ANN=0.05*rand(N_ANN,dim_ANN);  %input to hidden weights
Wli_ANN=0.05*rand(3,N_ANN);   %hidden to output weights
dWli_ANN(:,:,3)=zeros(3,N_ANN);
dWij_ANN(:,:,3)=zeros(N_ANN,dim_ANN);
%% Lyaponouv neural network initialization
N=4; % number of hidden neuron
dim=4;
Wij_d_c =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_d_c =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
W1j_d_c_new(:,:,2) = zeros(1,N);
Wij_d_c_new(:,:,2) = zeros(4,N);
Wij_d_e =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_d_e =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
W1j_d_e_new(:,:,2) = zeros(1,N);
Wij_d_e_new(:,:,2) = zeros(4,N);
Wij_p_c =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_p_c =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
W1j_p_c_new(:,:,2) = zeros(1,N);
Wij_p_c_new(:,:,2) = zeros(4,N);
Wij_p_e =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_p_e =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
W1j_p_e_new(:,:,2) = zeros(1,N);
Wij_p_e_new(:,:,2) = zeros(4,N);
Wij_i_c =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_i_c =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
W1j_i_c_new(:,:,2) = zeros(1,N);
Wij_i_c_new(:,:,2) = zeros(4,N);
Wij_i_e =0.05*rand(N,dim)*sqrt(2/(dim+N));  %input to hidden weights
W1j_i_e =0.05*rand(1,N)*sqrt(2/(1+N));   %hidden to output weights
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
Wv =170.2*pi;
Ch = 8.85*10^(-4);
Dv = 1;
Vqu = 0.1867;
T = 0.5;
B = 2/T;
alfa = (Vqu*Wv^2)/Ch;
K1= 3;%10^(-4)
K2= 4;
K3= 4;
a0 =  B^3 + (Wv^2 + alfa*K3)*B^2 + B*(2*Wv+alfa*K2) + alfa*K1;
a1 = -3*B^3 -(Wv^2+alfa*K3)*B^2 + 3*alfa*K1;
a2 = 3*B^3 -(Wv^2 +alfa*K3)*B^2 -B*(2*Wv^2+alfa*K2)+ 3*alfa*K1;
a3 = -B^3 + (Wv^2 +alfa*K3)*B^2 + alfa*K1;
% a4 = B^4 - 2*Dv*Wv*B^3 - 3*(Wv^2 + alfa*Kd)*B^2 - B*alfa*Kp +alfa*Ki;
b0 =alfa;
b1 = 3*alfa;
b2 = 3*alfa;
b3 = alfa;
% b4 = alfa*Ki - B*alfa*Kp;
err_p_c(4) = 0;
err_d_c(4) = 0;
y(1,1)=0;
y(2,1)=0;
y(3,1)=0;
y(4,1)=0;
e(1,1)=x(1)-y(1);
e(2,1)=x(2)-y(2);
e(3,1)=x(3)-y(3);
e(4,1)=x(4)-y(4);
u_p_c(1,1)=0;
u_p_c(2,1)=0;
u_p_c(3,1)=0;
u_p_c(4,1)=0;
u_p_e(4,1) = 0.05;
u_i_c(1,1)=0;
u_i_c(2,1)=0;
u_i_c(3,1)=0;
u_i_c(4,1)=0;
u_i_e(4,1) = 0.05;
u_d_c(1,1)=0;
u_d_c(2,1)=0;
u_d_c(3,1)=0;
u_d_c(4,1)=0;
u_d_e(4,1) = 0.05;
u(4,1) = 0;
u(3,1) = 0;
u(2,1) = 0;
Kp=0.1191;
Ki=0.9241;
Kd=0.0024;
S_p_c(3,:)= [0.1 0.2 0.3 0.4];
S_p_c(4,:)= [0.1 0.2 0.3 0.4];
S_p_e(3,:)= [0.1 0.2 0.3 0.4];
S_p_e(4,:)= [0.1 0.2 0.3 0.4];
S_i_c(3,:)= [0.1 0.2 0.3 0.4];
S_i_c(4,:)= [0.1 0.2 0.3 0.4];
S_i_e(3,:)= [0.1 0.2 0.3 0.4];
S_d_c(3,:)= [0.1 0.2 0.3 0.4];
S_d_e(3,:)= [0.1 0.2 0.3 0.4];
S_i_e(4,:)= [0.1 0.2 0.3 0.4];
S_d_c(4,:)= [0.1 0.2 0.3 0.4];
S_d_e(4,:)= [0.1 0.2 0.3 0.4];
Kp_pid = 0.3;
Ki_pid = 0.2;
Kd_pid = 0.01;
y_pid(1)=0;
y_pid(2)=0;
y_pid(3)=0;
e_pid(1)=r(1)-y_pid(1);
e_pid(2)=r(2)-y_pid(2);
e_pid(3)=r(3)-y_pid(3);
u_pid(1)=0;
u_pid(2)=0;
u_pid(3)=0;
Kp_ANN=0.3;
Ki_ANN=0.2;
Kd_ANN=0.01;
Bb=1; 
y_ANN(1)=0;
y_ANN(2)=0;
y_ANN(3)=0;
e_ANN(1)=r(1)-y_ANN(1);
e_ANN(2)=r(2)-y_ANN(2);
e_ANN(3)=r(3)-y_ANN(3);
u_ANN(1)=0;
u_ANN(2)=0;
u_ANN(3)=0;
%% Traditional PID******************************************************
for k=4:length(t)
     u_pid(k)= u_pid(k-1) + Kp_pid*(e_pid(k-1)-e_pid(k-2)) + Ki_pid*e_pid(k-1)+ Kd_pid*(e_pid(k-1)-2*e_pid(k-2)+e_pid(k-3)); %PID 
    y_pid(k) = u_pid(k)*b0/a0 + u_pid(k-1)*b1/a0 + u_pid(k-2)*b2/a0 + u_pid(k-3)*b3/a0 - y_pid(k-1)*a1/a0 - y_pid(k-2)*a2/a0 - y_pid(k-3)*a3/a0 ;%x(k-4)*b4/a0 - y(k-4)*a4/a0
  e_pid(k)=r(k)-y_pid(k);  
     if t(k-1)==1
        u_pid(k)=3.5; %first disu_pid(k)+0.5
    elseif t(k-1)==1.8
        u_pid(k)=4.3; %second disu_pid(k)+0.89
    end
    
end
%% ANN-PID**************************************************************
for k=4:length(t)

% plant & PID & error
    u_ANN(k)= u_ANN(k-1) + Kp_ANN*(e_ANN(k-1)-e_ANN(k-2)) + Ki_ANN*e_ANN(k-1)+ Kd_ANN*(e_ANN(k-1)-2*e_ANN(k-2)+e_ANN(k-3)); %PID 
    y_ANN(k) = u_ANN(k)*b0/a0 + u_ANN(k-1)*b1/a0 + u_ANN(k-2)*b2/a0 + u_ANN(k-3)*b3/a0 - y_ANN(k-1)*a1/a0 - y_ANN(k-2)*a2/a0 - y_ANN(k-3)*a3/a0 ;%x(k-4)*b4/a0 - y(k-4)*a4/a0
  e_ANN(k)=r(k)-y_ANN(k);   %error
    e2_ANN(k,1)=e_ANN(k)-e_ANN(k-1);
    loss_ANN(k) = 0.5*e_ANN(k)^2;
%     loss = mean(loss);
%% disturbance
    if t(k-1)==1
        u_ANN(k)=3.5; %first disu_ANN(k)+0.5
    elseif t(k-1)==1.8
        u_ANN(k)=4.3; %second disu_ANN(k)+0.89
    end
    %% neural network
    % input layer
    x_ANN=[r(k);u_ANN(k);e_ANN(k);e2_ANN(k,1)];  
    O1_ANN=x_ANN;
    
    %hidden layyer
    net2_ANN=Wij_ANN*O1_ANN;
%     O2=tanh(net2);
    O2_ANN =net2_ANN/(1 + exp(Bb*net2_ANN));
    
    %outout layer
    net3_ANN=Wli_ANN*O2_ANN;
    O3_ANN=0.5*(1+tanh(net3_ANN));
    
    
    %% updating weights between hidden and output layer
    for i=1:N_ANN
        for l=1:3
            activLJ_ANN=0.5+0.5*tanh(net3_ANN(l));
            gradactivLJ_ANN=activLJ_ANN*(1-activLJ_ANN);
%        Sig = 1/(1+exp(B*net3(l)));
%        activLJ = net3(l)/(1 + exp(B*net3(l)));    % SWISH activation function
%        gradactivLJ = B*activLJ + Sig*(1-B*activLJ);  %gradient of activation function
            if l==1
                Delta3_ANN(l)=e_ANN(k)*sign(gradient(y_ANN(k),u_ANN(k)))*(e_ANN(k)-e_ANN(k-1))*gradactivLJ_ANN;
            elseif l==2
                Delta3_ANN(l)=e_ANN(k)*sign(gradient(y_ANN(k),u_ANN(k)))*(e_ANN(k))*gradactivLJ_ANN;
            elseif l==3
                Delta3_ANN(l)=e_ANN(k)*sign(gradient(y_ANN(k),u_ANN(k)))*(e_ANN(k)-2*e_ANN(k-1)+e_ANN(k-2))*gradactivLJ_ANN;
            end
            
            dWli_ANN(l,i,k)=alf* dWli_ANN(l,i,k-1)+eta_ANN*Delta3_ANN(l)*O2_ANN(i);
            Wli_new_ANN(l,i)=Wli_ANN(l,i)+dWli_ANN(l,i,k);
        end
    end
    %% updating weights between input and hidden layer
    for i=1:N_ANN
        for j=1:dim_ANN
%             f=tanh(net2(i)); 
%             fp=0.5-0.5*f*f;
             Sig_ANN = 1/(1+exp(Bb*net2_ANN(i)));
             activIJ_ANN = net2_ANN(i)/(1 + exp(Bb*net2_ANN(i)));    % SWISH activation function
             gradactivIJ_ANN = Bb*activIJ_ANN + Sig_ANN*(1-Bb*activIJ_ANN);  %gradient of activation function
            for l=1:3
                S_ANN(l)=Delta3_ANN(l)*Wli_ANN(l,i);
            end
            Delta2_ANN(i)=gradactivIJ_ANN*sum(S_ANN);
            
            dWij_ANN(i,j,k)=alf*dWij_ANN(i,j,k-1)+eta_ANN*Delta2_ANN(i)*O1_ANN(j);
            Wij_ANN(i,j)=Wij_ANN(i,j)+dWij_ANN(i,j,k);
        end
    end
    
    Wli_ANN=Wli_new_ANN;
                
    net2_ANN=Wij_ANN*O1_ANN;
    O2_ANN=tanh(net2_ANN);
    
    net3_ANN=Wli_ANN*O2_ANN;
    O3new_ANN=0.5*(1+tanh(net3_ANN));
    K_ANN(:,k)=O3new_ANN;
%     O3 = O3new;

    Kp_ANN=O3new_ANN(1)/3.33;
    Ki_ANN=O3new_ANN(2)/3.33;
    Kd_ANN=O3new_ANN(3)/20;
    K_ANN(1,k)=Kp_ANN;
    K_ANN(2,k)=Ki_ANN;
    K_ANN(3,k)=Kd_ANN;
    k = k+1;
end
%% Lyaponouv-PID **************************************************
for k = 5: length(t)
  k
 
  %% plant model

   if t(k-1)==1
        u(k)=3.5; %first disu(k)+0.5
    elseif t(k-1)==1.8
        u(k)=4.3; %second dis
   else
u(k)= u(k-1) + Kp*(e(k-1)-e(k-2)) + Ki*e(k-1)+ Kd*(e(k-1)-2*e(k-2)+e(k-3));
 end
y(k) = u(k)*b0/a0 + u(k-1)*b1/a0 + u(k-2)*b2/a0 + u(k-3)*b3/a0 - y(k-1)*a1/a0 - y(k-2)*a2/a0 - y(k-3)*a3/a0 ;%x(k-4)*b4/a0 - y(k-4)*a4/a0
% 
e(k)=x(k)-y(k);
% sys z
% mish(z) = z*tanh(ln(1+exp(z)));
% mish_inv(z) = finverse(mish(z));

%% NNP
err_p_c(k) = e(k);
%NNC*****************************************************************
   for j = 1:4
   for i= 1:4
       S_p_c(k,j) = tanh(Wij_p_c(j,i).*x_p_c(k-i+1));
%  S_p_c(k,j) = Wij_p_c(j,i).*x_p_c(k-i+1);
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
u_p_c(k) =0.5*(1+ tanh(net3_p_c(1,1)+ net3_p_c(1,2) + net3_p_c(1,3)+net3_p_c(1,4)));


% updating weights %%%%%%%%%
% Updating W1j
d_p_c(k)= u_p_e(k-1); 
sigma_p_c (k)  = sqrt(B_p_c)*err_p_c(k-1) + d_p_c(k);
for j = 1 :4
    W1j_p_c_new(k,j) = sigma_p_c (k) /(4*(S_p_c(k-1,j))); %changed it to mean spc
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_c_new(j,i) = (1/(4*x_p_c(k-i+1)))*atanh(sigma_p_c(k)/(4*W1j_p_c(1,j))); % Not OK
%  Wij_p_c_new(j,i) = (1/(4*x_p_c(k-i+1)))*finverse(mish(sigma_p_c(k)/(4*W1j_p_c(1,j))));
    end
end
Wij_p_c = Wij_p_c_new(:,:,1);
W1j_p_c = W1j_p_c_new(k,:,1); %first 3 rows are zero

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
    W1j_p_e_new(k,j) = sigma_p_e (k) /(4*(S_p_e(k-1,j))); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_p_e_new(j,i) = 1/(4*x_p_e(k-i+1))*atanh(sigma_p_e(k)/(4*W1j_p_e(1,j))); % Not OK
    end

end
Wij_p_e = Wij_p_e_new(:,:,1);
W1j_p_e = W1j_p_e_new(k,:,1); % only the 4th row has positive num


%% NNI
%NNC *****************************************************************

err_i_c_sum  =  0;     %int(e(k));
for i = 1:k
    err_i_c_summ = err_i_c_sum + e(i);
end
err_i_c(k) = err_i_c_summ;
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
u_i_c(k) = 0.5*(1+tanh(net3_i_c(1,1)+ net3_i_c(1,2) + net3_i_c(1,3)+net3_i_c(1,4)));

err_i_c(k) = u_i_c(k) -x(k); %?????????????

% updating weights 
% Updating W1j
d_i_c(k)= u_i_e(k-1); % must be changed ******************u_p_e(k-1)
sigma_i_c (k)  = sqrt(B_i_c)*err_i_c(k-1) + d_i_c(k);
for j = 1 :4
    W1j_i_c_new(k,j) = sigma_i_c (k) /(4*(S_i_c(k-1,j))); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_i_c_new(j,i) = 1/(4*x_i_c(k-i+1))*atanh(sigma_i_c(k)/(4*W1j_i_c(1,j))); % Not OK
    end
end
Wij_i_c = Wij_i_c_new(:,:,1);
W1j_i_c = W1j_i_c_new(k,:,1);

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
    W1j_i_e_new(k,j) = sigma_i_e (k) /(4*(S_i_e(k-1,j))); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_i_e_new(j,i) = 1/(4*x_i_e(k-i+1))*atanh(sigma_i_e(k)/(4*W1j_i_e(1,j))); % Not OK
    end

end
Wij_i_e = Wij_i_e_new(:,:,1);
W1j_i_e = W1j_i_e_new(k,:,1);


%% NND
%NNC*****************************************************************
x_d_c (k) = x(k);
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
u_d_c(k) = 0.5*(1+tanh(net3_d_c(1,1)+ net3_d_c(1,2) + net3_d_c(1,3)+net3_d_c(1,4)));

% updating weights %%%%%%%%%
% Updating W1j
d_d_c(k)= u_d_e(k-1); % must be changed ******************u_p_e(k-1)
sigma_d_c (k)  = sqrt(B_d_c)*err_d_c(k-1) + d_d_c(k);
for j = 1 :4
    W1j_d_c_new(k,j) = sigma_d_c (k) /(4*S_d_c(k-1,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_d_c_new(j,i) = 1/(4*x_d_c(k-i+1))*atanh(sigma_d_c(k)/(4*W1j_d_c(1,j))); % Not OK
    end
end
Wij_d_c = Wij_d_c_new(:,:,1);
W1j_d_c = W1j_d_c_new(k,:,1);

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
    W1j_d_e_new(k,j) = sigma_d_e (k) /(4*S_d_e(k-1,j)); %changed
end

%Updating Wij
for i = 1:4
    for j = 1:4
        Wij_d_e_new(j,i) = 1/(4*x_d_e(k-i+1))*atanh(sigma_d_e(k)/(4*W1j_d_e(1,j))); % Not OK
    end

end
Wij_d_e = Wij_d_e_new(:,:,1);
W1j_d_e = W1j_d_e_new(k,:,1);
%end of nn

if e(k) >= 1e-12
Kp = abs(u_p_c(k)/3.33);
Ki = abs(u_i_c(k)/3.33);
Kd = abs(u_d_c(k)/20);
 K(k,1) = Kp;
 K(k,2) = Ki;
 K(k,3) = Kd;
else
    Kp = K(k-1,1);
    Ki=K(k-1,2);
    Kd=K(k-1,3);
     K(k,1) = Kp;
 K(k,2) = Ki;
 K(k,3) = Kd;

end
    
end
%
figure('Name','System Output','NumberTitle','off');
plot(t,y_pid,'LineWidth',1);
grid on
set(gca,'FontSize',12)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y(k)');
title('System Output');
hold on
plot(t,y_ANN,'LineWidth',1);
hold on
plot(t,y,'g','LineWidth',1);
hold off
legend('y_{PID}(k)','y_{NN-PID}(k)','y_{lyapanuv-PID}')
%

figure('Name','system output(0.5 second)','NumberTitle','off');
plot(t,y_pid,'LineWidth',1.5);
grid on
set(gca,'FontSize',12)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y(k)');
axis([0 0.5 0 1.5])
title('system output(0.5 second)');
hold on
plot(t,y_ANN,'LineWidth',1.5);
hold on
plot(t,y,'g','LineWidth',1.5);
hold off
legend('y_{PID}(k)','y_{NN-PID}(k)','y_{lyapanuv-PID}')

%system errors
figure('Name','System error','NumberTitle','off');
plot(t,abs(e_pid),'LineWidth',1);
grid on
set(gca,'FontSize',12)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y(k)');
title('System Error');
hold on
plot(t,abs(e_ANN),'LineWidth',1);
hold on
plot(t,abs(e),'g','LineWidth',1);
hold off
legend('e_{PID}(k)','e_{NN-PID}(k)','e_{lyapanuv-PID}')

%control signals
figure('Name','Control signal','NumberTitle','off');
plot(t,u_pid,'LineWidth',1);
grid on
set(gca,'FontSize',12)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y(k)');
title('Control signal');
hold on
plot(t,u_ANN,'LineWidth',1);
hold on
plot(t,u,'g','LineWidth',1);
hold off
legend('e_{PID}(k)','e_{NN-PID}(k)','e_{lyapanuv-PID}')
%
figure
plot(t,u,'r','LineWidth',1.5)
xlabel('time(s)')
ylabel('u(k)')
title('control signal')
legend('u(k)')
axis([0 3.5 0 5])
grid on

figure
plot(t,e,'r','LineWidth',1.5)
xlabel('time(s)')
ylabel('e(k)')
title('system error')
axis([0 3.5 -1 1])
grid on

figure
plot(t,y,'r','LineWidth',1.5)
xlabel('time(s)')
ylabel('Y(k)')
title('system output(0.5 second)')
axis([0 0.5 0 2])
grid on

figure
plot(t,u,'r','LineWidth',1.5)
xlabel('time(s)')
ylabel('u(k)')
title('control signal(0.5 second)')
legend('u(k)')
axis([0 0.5 0 5])
grid on

figure
plot(t,e,'r','LineWidth',1.5)
xlabel('time(s)')
ylabel('e(k)')
title('system error(0.5 second)')
axis([0 0.5 -1 1])
grid on
