clear;close all;clc


modelname='C:\资料\vp-experiments\model_svp_is_run2\\';
conicname='C:\资料\vp-experiments\model_newsft2_run1\';
% show=1;
show=0;

for iii=100:-1:91
%     a=[modelname,'/distgt_',num2str(iii),'.mat'];
%     b=[modelname,'/errgt_',num2str(iii),'.mat'];

    a=[modelname,'/dist_',num2str(iii),'.mat'];
    b=[modelname,'/err_',num2str(iii),'.mat'];
    c=[conicname,'/dist_',num2str(iii),'.mat'];
    d=[conicname,'/err_',num2str(iii),'.mat'];

%     a=[modelname,'/dist_',num2str(92),'.mat'];
%     b=[modelname,'/err_',num2str(92),'.mat'];
%     c=[conicname,'/dist_',num2str(95),'.mat'];
%     d=[conicname,'/err_',num2str(95),'.mat'];
    
%     a=[modelname,'/dist-2_',num2str(92),'.mat'];
%     b=[modelname,'/err-2_',num2str(92),'.mat'];
%     c=[conicname,'/dist-2_',num2str(95),'.mat'];
%     d=[conicname,'/err-2_',num2str(95),'.mat'];
    
%     index=[2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,23,24,25,27,29,30,31,33,34,36];
    dist=load (a);
    dist=dist.name;
    index=[1:length(dist)];
    distance=dist(index);
    maxdist=1;
    if max(distance)<10
        maxdist=max(distance);
    end
%     xx = 0:0.0001:1;threshold=0:0.00001:1;
    xx = 0:0.0001:maxdist;threshold=0:0.00001:maxdist;
    if show
%         figure;grid on;hold on;xlabel('Distance');ylabel('Percentage');xlim([0,1]);ylim([0,0.75]);
        figure;grid on;hold on;xlabel('Distance');ylabel('Percentage');xlim([0,maxdist]);ylim([0,1]);
    end
    Ours=[];
    NeurVPS=[];
    for i=1:length(threshold)
        Ours(i)=length(find((distance)<threshold(i)))/length(distance);
    end
    if show
        yy=spline(threshold,[0 Ours 0],xx);Ours=plot(xx,yy,'LineWidth',2);
    end
    fprintf('model1\n mean median 0.04 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.04)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))

%     fprintf('Ours\n mean median 0.04 0.1 0.15 0.2 0.25 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),sum(distance<0.04)/size(distance,2),sum(distance<0.1)/size(distance,2),sum(distance<0.15)/size(distance,2),sum(distance<0.2)/size(distance,2),sum(distance<0.25)/size(distance,2),sum(distance<0.3)/size(distance,2))
    
    dist=load (c);
    dist=dist.name;
    index=[1:length(dist)];
    distance=dist(index);
    for i=1:length(threshold)
        NeurVPS(i)=length(find(distance<threshold(i)))/length(distance);
    end
    fprintf('model2\n mean median 0.04 0.1 0.15 0.2 0.25 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),sum(distance<0.04)/size(distance,2),sum(distance<0.1)/size(distance,2),sum(distance<0.15)/size(distance,2),sum(distance<0.2)/size(distance,2),sum(distance<0.25)/size(distance,2),sum(distance<0.3)/size(distance,2))
    if show
        yy=spline(threshold,[0 NeurVPS 0],xx);NeurVPS=plot(xx,yy,'LineWidth',2);legend([Ours NeurVPS],'Ours','NeurVPS');  
    end

    err=load (b);
    angles=err.name;
    angle=angles(index);
    maxerr=1;
    if max(angle)<100
        maxerr=max(angle);
    end
    if show
        xx = 0:0.01:max(maxerr);threshold=0:0.0001:max(maxerr);figure;grid on;hold on;xlabel('Angle Difference');ylabel('Percentage');xlim([0,maxerr]);ylim([0,1]);
    end
    Ours=[];
    NeurVPS=[];
    for i=1:length(threshold)
        Ours(i)=length(find((angle)<threshold(i)))/length(angle) ;
    end
    fprintf('model1\n mean median 1 2 5 10 \n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))
    if show
        yy=spline(threshold,[0 Ours 0],xx);Ours=plot(xx,yy,'LineWidth',2);
    end
    err=load (d);
    angles=err.name;
    angle=angles(index);
    for i=1:length(threshold)
        NeurVPS(i)=length(find(angle<threshold(i)))/length(angle);
    end
    fprintf('model2\n mean median 1 2 5 10 20\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),sum(angle<1)/size(angle,2),sum(angle<2)/size(angle,2),sum(angle<5)/size(angle,2),sum(angle<10)/size(angle,2),sum(angle<20)/size(angle,2))
    if show
        yy=spline(threshold,[0 NeurVPS 0],xx);NeurVPS=plot(xx,yy,'LineWidth',2);legend([Ours NeurVPS],'Ours','NeurVPS'); 

    end
    disp(iii)
%             break;

end