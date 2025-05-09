clear;close all;clc

modelname='C:\资料\vp-experiments\model_detection_sft_is_data3\';
conicname='C:\资料\vp-experiments\model_vpnet_data3\';
% show=1;
show=0;

bestmodel=92;
bestconic=95;

a=[modelname,'/data3all_',num2str(bestmodel),'.mat'];
imn=load (a);
imname=imn.name;
imname=str2num(squeeze(imname(:,:,62-7:62-4)));

a=[modelname,'/data3_',num2str(bestmodel),'.mat'];
imn=load (a);
imname2=imn.name;
imname2=str2num(squeeze(imname2(:,:,68-7:68-4)));

[tf2,]=ismember(imname,imname2);
index2=find(tf2~=0);
% 
% 
index=index2;
% index=[1:111]
% 
% 

modelname='C:\资料\vp-experiments\model_detection_sft_is_run3_18_data3\';
conicname='C:\资料\vp-experiments\model_vpnet_data3test\';
bestmodel=100;
bestconic=100;
a=[modelname,'/dist_',num2str(bestmodel),'.mat'];
b=[modelname,'/err_',num2str(bestmodel),'.mat'];
%     a=[modelname,'/dist_',num2str(bestmodel),'.mat'];
%     b=[modelname,'/err_',num2str(bestmodel),'.mat'];
c=[conicname,'/dist_',num2str(bestconic),'.mat'];
d=[conicname,'/err_',num2str(bestconic),'.mat'];


dist=load (a);
distance=dist.name;
distance=distance(index);
maxdist=1;
if max(distance)<10
    maxdist=max(distance);
end
xx = 0:0.0001:maxdist;threshold=0:0.00001:maxdist;
if show
    figure;grid on;hold on;
    
    xlabel('Focal distance difference','FontSize',20,'FontName','times new roman');
    % 创建 ylabel
    ylabel('Percentage','FontSize',20,'FontName','times new roman');
    xlim([0,maxdist]);ylim([0,1]);
    title('FDA Curve for the SVPI dataset') 
end
Ours=[];
NeurVPS=[];
for i=1:length(threshold)
    Ours(i)=length(find((distance)<threshold(i)))/length(distance);
end
if show
    yy=spline(threshold,[0 Ours 0],xx);Ours=plot(xx,yy,'LineWidth',2);
end
fprintf('Ours\n mean median 0.04 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.04)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))

%     fprintf('Ours\n mean median 0.04 0.1 0.15 0.2 0.25 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),sum(distance<0.04)/size(distance,2),sum(distance<0.1)/size(distance,2),sum(distance<0.15)/size(distance,2),sum(distance<0.2)/size(distance,2),sum(distance<0.25)/size(distance,2),sum(distance<0.3)/size(distance,2))

dist=load (c);
distance=dist.name;
distance=distance(index);
for i=1:length(threshold)
    NeurVPS(i)=length(find(distance<threshold(i)))/length(distance);
end
fprintf('NeurVPS\n mean median 0.04 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.04)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))
if show
    yy=spline(threshold,[0 NeurVPS 0],xx);NeurVPS=plot(xx,yy,'LineWidth',2);legend1=legend([Ours NeurVPS],'Ours','NeurVPS');  
    set(legend1,'FontSize',16);
end


err=load (b);
angle=err.name;
angle=angle(index);
maxerr=1;
if max(angle)<100
    maxerr=max(angle);
end
if show
    xx = 0:0.01:max(maxerr);threshold=0:0.0001:max(maxerr);figure;grid on;hold on;
    xlabel('Angle difference','FontSize',20,'FontName','times new roman');
    % 创建 ylabel
    ylabel('Percentage','FontSize',20,'FontName','times new roman');
    xlim([0,maxerr]);ylim([0,1]);title('AA Curve for the SVPI dataset')
end
Ours=[];
NeurVPS=[];
for i=1:length(threshold)
    Ours(i)=length(find((angle)<threshold(i)))/length(angle) ;
end
fprintf('Ours\n mean median 1 2 5 10 \n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))
if show
    yy=spline(threshold,[0 Ours 0],xx);Ours=plot(xx,yy,'LineWidth',2);
end
err=load (d);
angle=err.name;
angle=angle(index);
for i=1:length(threshold)
    NeurVPS(i)=length(find(angle<threshold(i)))/length(angle);
end
fprintf('NeurVPS\n mean median 1 2 5 10\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))
if show
    yy=spline(threshold,[0 NeurVPS 0],xx);NeurVPS=plot(xx,yy,'LineWidth',2);legend2=legend([Ours NeurVPS],'Ours','NeurVPS');  
    set(legend2,'FontSize',16);
end