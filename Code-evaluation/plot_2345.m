clear;close all;clc

modelname='C:\资料\vp-experiments\model_detection_sft_is_run3_18\';
conicname='C:\资料\vp-experiments\model_vpnet\';
% show=1;
show=0;

bestmodel=92;
bestconic=95;

a=[modelname,'/image_',num2str(bestmodel),'.mat'];
imn=load (a);
imname=imn.name;
imname=str2num(squeeze(imname(:,:,62-7:62-4)));

a=[modelname,'/image2_',num2str(bestmodel),'.mat'];
imn=load (a);
imname2=imn.name;
imname2=str2num(squeeze(imname2(:,:,63-7:63-4)));

a=[modelname,'/image3_',num2str(bestmodel),'.mat'];
imn=load (a);
imname3=imn.name;
imname3=str2num(squeeze(imname3(:,:,63-7:63-4)));

a=[modelname,'/image4_',num2str(bestmodel),'.mat'];
imn=load (a);
imname4=imn.name;
imname4=str2num(squeeze(imname4(:,:,63-7:63-4)));

a=[modelname,'/image5_',num2str(bestmodel),'.mat'];
imn=load (a);
imname5=imn.name;
imname5=str2num(squeeze(imname5(:,:,63-7:63-4)));

a=[modelname,'/image5p_',num2str(bestmodel),'.mat'];
imn=load (a);
imname5p=imn.name;
imname5p=str2num(squeeze(imname5p(:,:,64-7:64-4)));

a=[modelname,'/image_overlap_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap=imn.name;
imnameoverlap=str2num(squeeze(imnameoverlap(:,:,70-7:70-4)));

a=[modelname,'/image_overlap2_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap2=imn.name;
imnameoverlap2=str2num(squeeze(imnameoverlap2(:,:,71-7:71-4)));

a=[modelname,'/image_overlap3_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap3=imn.name;
imnameoverlap3=str2num(squeeze(imnameoverlap3(:,:,71-7:71-4)));

a=[modelname,'/image_overlap4_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap4=imn.name;
imnameoverlap4=str2num(squeeze(imnameoverlap4(:,:,71-7:71-4)));

a=[modelname,'/image_overlap5_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap5=imn.name;
imnameoverlap5=str2num(squeeze(imnameoverlap5(:,:,71-7:71-4)));

a=[modelname,'/image_overlap5p_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap5p=imn.name;
imnameoverlap5p=str2num(squeeze(imnameoverlap5p(:,:,72-7:72-4)));


[tf2,]=ismember(imname,imname2);
[tf3,]=ismember(imname,imname3);
[tf4,]=ismember(imname,imname4);
[tf5,]=ismember(imname,imname5);
[tf5p,]=ismember(imname,imname5p);
[tfoverlap,]=ismember(imname,imnameoverlap);
[tfoverlap2,]=ismember(imname,imnameoverlap2);
[tfoverlap3,]=ismember(imname,imnameoverlap3);
[tfoverlap4,]=ismember(imname,imnameoverlap4);
[tfoverlap5,]=ismember(imname,imnameoverlap5);
[tfoverlap5p,]=ismember(imname,imnameoverlap5p);
index2=find(tf2~=0);
index3=find(tf3~=0);
index4=find(tf4~=0);
index5=find(tf5~=0);
index5p=find(tf5p~=0);
indexoverlap=find(tfoverlap~=0);
indexoverlap2=find(tfoverlap2~=0);
indexoverlap3=find(tfoverlap3~=0);
indexoverlap4=find(tfoverlap4~=0);
indexoverlap5=find(tfoverlap5~=0);
indexoverlap5p=find(tfoverlap5p~=0);
% 
% 
index=indexoverlap3;
% index=[1:111]
% 
% 

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