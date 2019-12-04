close all;clear all;clc;
R1=1;
C1=0;
XY=importdata('whitewine.xlsx');
mat=XY.data;
n=length(mat);
X=mat(:,1:11);
Y=mat(:,12);
x_proc=[mean(X(:,1)) mean(X(:,2)) mean(X(:,3)) mean(X(:,4)) mean(X(:,5)) mean(X(:,6)) mean(X(:,7)) mean(X(:,8)) mean(X(:,9)) mean(X(:,10)) mean(X(:,11))];
tau=0.1:0.5:10;
val=10;
error_train_vek=zeros(1,length(tau));
error_test_vek=zeros(1,length(tau));
for i=1:length(tau)
    t=tau(i);
    error_train=0;
    error_test=0;
   for j=1:val
    x_train=[X(1:(j-1)*floor(n/val),:);X(j*floor(n/val)+1:end,:)];
    x_test=X((j-1)*floor(n/val)+1:j*floor(n/val),:);
    y_train= [Y(1:(j-1)*floor(n/val),:);Y(j*floor(n/val)+1:end,:)];
    y_test= Y((j-1)*floor(n/val)+1:j*floor(n/val),:);
    Wt=zeros(length(x_train),length(x_train));
    for k=1: length(x_train)
        Wt(k,k)=exp(-(norm(x_train(k,:)-x_proc))^2/(2*t^2));
    end
    Wtt=zeros(length(x_test),length(x_test));
    for k=1:length(x_test)
         Wtt(k,k)= exp(-(norm(x_test(k,:)-x_proc))^2/(2*t^2));
    end
    theta=(x_train'*Wt*x_train)^(-1)*(x_train'*Wt*y_train);
    error_train=(y_train-x_train*theta)'*(y_train-x_train*theta);
    error_test=(y_test-x_test*theta)'*(y_test-x_test*theta);
    error_train=error_train+error_train;
    error_test=error_test+error_test;
    end
    error_train_vek(i)=error_train/val;
    error_test_vek(i)=error_test/val;
end
[~,ind]=min(error_test_vek);
minn=tau(ind);
figure(3);
plot(tau,error_train_vek,'r');
hold on;
plot(tau,error_test_vek); 
legend('train','test');
grid on;
xlabel('tau');
ylabel('greska');
W1=zeros(4000,4000);
X_train=X(1:4000,:);
Y_train=Y(1:4000,:);
X_test=X(4001:4898,:);
Y_test=Y(4001:4898,:);
for k=1:4000
    W1(k,k)=exp(-(norm(X_train(k,:)-x_proc))^2/(2*10^2));
end  
theta=(X_train'*W1*X_train)^(-1)*X_train'*W1*Y_train; 
mat_con1=zeros(9,9);
for i=1:898
     pred=round(X_test(i,:)*theta);
     mat_con1(Y_test(i),pred)= mat_con1(Y_test(i),pred)+1;
end
 y_forecast_3=sum(mat_con1(:,3));
 y_forecast_4=sum(mat_con1(:,4));
 y_forecast_5=sum(mat_con1(:,5));
 y_forecast_6=sum(mat_con1(:,6));
 y_forecast_7=sum(mat_con1(:,7));
 y_forecast_8=sum(mat_con1(:,8));
 y_forecast_9=sum(mat_con1(:,9));
 y_forecasts=[y_forecast_3 y_forecast_4 y_forecast_5 y_forecast_6 y_forecast_7 y_forecast_8 y_forecast_9 ];
 x_forecasts=[3 4 5 6 7 8 9];
 figure(10);
 plot(x_forecasts,y_forecasts,'b*','MarkerSize',20);
 hold all;
 correct=0;
 underestimated=0; 
 overestimated=0;
 for i=1:9
    for j=1:9 
     if(i==j)correct=correct+mat_kon1(i,i);
     end
     if(i<j) underestimated=underestimated+mat_kon1(i,j);
     end
     if(i>j) overestimated=overestimated+mat_kon1(i,j);
     end
    end
 end

 
    
    
    
    
    
