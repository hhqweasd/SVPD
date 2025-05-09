clear;close all;clc

model1name='C:\资料\vp-experiments\model_svp_is_run2\';


model2name='C:\资料\vp-experiments\model_vpnet_data3_run1\';
% model2name='C:\资料\vp-experiments\model_svp_run2\';
% model2name='C:\资料\vp-experiments\model_svp_is_abl_wosft_run1\';
% model2name='C:\资料\vp-experiments\model_svp_is_abl_wop_run1\';
% model2name='C:\资料\vp-experiments\model_svp_is_abl_sft_run7\';
% model2name='C:\资料\vp-experiments\model_svp_is_abl_sft_run7\';

show=1;
% show=0;

for iii=100:-1:59
% for iii=77:1:100
%     iii=77
%     a=[model1name,'/dist_',num2str(77),'.mat'];
%     b=[model1name,'/err_',num2str(77),'.mat'];
    a=[model1name,'/dist.mat'];
    b=[model1name,'/err.mat'];
    
%     iii=100
    c=[model2name,'/dist_',num2str(iii),'.mat'];
    d=[model2name,'/err_',num2str(iii),'.mat'];
    

    
    index=[2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,23,24,25,27,29,30,31,33,34,36];


    %% FDA
    dist=load(a);
    dist=dist.name;
    index=[1:length(dist)];
    distance=dist(index);
    maxdist=1;
    if max(distance)<10
        maxdist=max(distance);
    end
    xx = 0:0.0001:maxdist;threshold=0:0.00001:maxdist;
    Ours=[];
    NeurVPS=[];
    for i=1:length(threshold)
        Ours(i)=length(find((distance)<threshold(i)))/length(distance);
    end
    fprintf('model1\n mean median 0.05 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.05)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))

    dist=load(c);
    dist=dist.name;
    distance=dist(index);
    for i=1:length(threshold)
        NeurVPS(i)=length(find(distance<threshold(i)))/length(distance);
    end
    fprintf('model2\n mean median 0.05 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.05)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))
    if show
        figure;grid on;hold on;box on;
        xlabel('RD','FontName','Times New Roman');
        ylabel('Percentage','FontName','Times New Roman');
%         xlim([0,maxdist]);
        xlim([0,0.35]);
        ylim([0,1]);
        yy=spline(threshold,[0 Ours 0],xx);
        Ours=plot(xx,yy,'LineWidth',2);
        set(Ours,'DisplayName','Ours','Color',[1 0.600000023841858 0.7843137383461]);
        yy=spline(threshold,[0 NeurVPS 0],xx);
        NeurVPS=plot(xx,yy,'LineWidth',2);
        set(NeurVPS,'DisplayName','NeurVPS','LineStyle','--','Color',[0.23137255012989 0.443137258291245 0.337254911661148]);
        lg=legend([Ours NeurVPS],'Ours','NeurVPS');  
        set(lg,'Location','southeast','FontSize',14);
        set(gca,'FontName','Times New Roman','FontSize',14,'XTick',...
        [0 0.05 0.1 0.15 0.2 0.25 0.3 0.35],'YTick',[0 0.25 0.5 0.75 1],'YTickLabel',...
        {'0%','25%','50%','75%','100%'});
    end    
    
    %% AA
    err=load(b);
    err=err.name;
    angles=err;
    angle=angles(index);
    maxerr=1;
    if max(angle)<100
        maxerr=max(angle);
    end
    xx = 0:0.01:max(maxerr);
    threshold=0:0.0001:max(maxerr);
    Ours=[];
    NeurVPS=[];
    for i=1:length(threshold)
        Ours(i)=length(find((angle)<threshold(i)))/length(angle) ;
    end
    fprintf('model1\n mean median 1 2 5 10 \n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))
    
    err=load(d);
    err=err.name;
    angles=err;
    angle=angles(index);
    for i=1:length(threshold)
        NeurVPS(i)=length(find(angle<threshold(i)))/length(angle);
    end
    fprintf('model2\n mean median 1 2 5 10 \n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))
    if show
        figure;grid on;hold on;box on;
        xlabel('AD','FontName','Times New Roman');
        ylabel('Percentage','FontName','Times New Roman');
        xlim([0,15]);
        ylim([0,1]);
        yy=spline(threshold,[0 Ours 0],xx);
        Ours=plot(xx,yy,'LineWidth',2);
        set(Ours,'DisplayName','Ours','Color',[1 0.600000023841858 0.7843137383461]);
        yy=spline(threshold,[0 NeurVPS 0],xx);
        NeurVPS=plot(xx,yy,'LineWidth',2);
        set(NeurVPS,'DisplayName','NeurVPS','LineStyle','--','Color',[0.23137255012989 0.443137258291245 0.337254911661148]);
        lg=legend([Ours NeurVPS],'Ours','NeurVPS');  
        set(lg,'Location','southeast','FontSize',14);
        set(gca,'FontName','Times New Roman','FontSize',14,'XGrid','on','XTick',...
        [0 3 6 9 12 15],'XTickLabel',{'0°','3°','6°','9°','12°','15°'},'YGrid',...
        'on','YTick',[0 0.25 0.5 0.75 1],'YTickLabel',...
        {'0%','25%','50%','75%','100%'});
    end  

    disp(iii)
    break;

end