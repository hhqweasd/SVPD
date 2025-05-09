clear;close all;clc

modelname='C:\资料\vp-experiments\model_detection_sft_is_run3_18\';
conicname='C:\资料\vp-experiments\model_detection_sft_is_run3_18_data3\';
% % show=1;
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

a=[conicname,'/data3_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap=imn.name;
imnameoverlap=str2num(squeeze(imnameoverlap(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));

a=[conicname,'/data32_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap2=imn.name;
imnameoverlap2=str2num(squeeze(imnameoverlap2(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));

a=[conicname,'/data33_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap3=imn.name;
imnameoverlap3=str2num(squeeze(imnameoverlap3(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));

a=[conicname,'/data34_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap4=imn.name;
imnameoverlap4=str2num(squeeze(imnameoverlap4(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));

a=[conicname,'/data35_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap5=imn.name;
imnameoverlap5=str2num(squeeze(imnameoverlap5(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));

a=[conicname,'/data35p_',num2str(bestmodel),'.mat'];
imn=load (a);
imnameoverlap5p=imn.name;
imnameoverlap5p=str2num(squeeze(imnameoverlap5p(:,:,size(imn.name,3)-7:size(imn.name,3)-4)));


[tf2,]=ismember(imname,imname2);
[tf3,]=ismember(imname,imname3);
[tf4,]=ismember(imname,imname4);
[tf5,]=ismember(imname,imname5);
[tf5p,]=ismember(imname,imname5p);
[tfoverlap2,]=ismember(imnameoverlap,imnameoverlap2);
[tfoverlap3,]=ismember(imnameoverlap,imnameoverlap3);
[tfoverlap4,]=ismember(imnameoverlap,imnameoverlap4);
[tfoverlap5,]=ismember(imnameoverlap,imnameoverlap5);
[tfoverlap5p,]=ismember(imnameoverlap,imnameoverlap5p);
index2=find(tf2~=0);
index3=find(tf3~=0);
index4=find(tf4~=0);
index5=find(tf5~=0);
index5p=find(tf5p~=0);
indexoverlap2=find(tfoverlap2~=0);
indexoverlap3=find(tfoverlap3~=0);
indexoverlap4=find(tfoverlap4~=0);
indexoverlap5=find(tfoverlap5~=0);
indexoverlap5p=find(tfoverlap5p~=0);
% 
% 
indexdata12=index2;
indexdata3=indexoverlap2;
% index=[1:111]
% 
% 

modelname='C:\资料\vp-experiments\model_vpnet\';
conicname='C:\资料\vp-experiments\model_vpnet_data3all\';
bestmodel=95;

a=[modelname,'/dist_',num2str(bestmodel),'.mat'];
b=[modelname,'/err_',num2str(bestmodel),'.mat'];
%     a=[modelname,'/dist_',num2str(bestmodel),'.mat'];
%     b=[modelname,'/err_',num2str(bestmodel),'.mat'];
c=[conicname,'/dist_',num2str(bestmodel),'.mat'];
d=[conicname,'/err_',num2str(bestmodel),'.mat'];


dist1=load (a);
distance1=dist1.name;
distance1=distance1(indexdata12);
dist2=load (c);
distance2=dist2.name;
distance2=distance2(indexdata3);

% distance=distance1;
distance=[distance1 distance2];

fprintf('Ours\n mean median 0.04 0.1 0.2 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),100*sum(distance<0.04)/size(distance,2),100*sum(distance<0.1)/size(distance,2),100*sum(distance<0.2)/size(distance,2),100*sum(distance<0.3)/size(distance,2))

%     fprintf('Ours\n mean median 0.04 0.1 0.15 0.2 0.25 0.3\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(distance),median(distance),sum(distance<0.04)/size(distance,2),sum(distance<0.1)/size(distance,2),sum(distance<0.15)/size(distance,2),sum(distance<0.2)/size(distance,2),sum(distance<0.25)/size(distance,2),sum(distance<0.3)/size(distance,2))

err1=load (b);
angle1=err1.name;
angle1=angle1(indexdata12);
err2=load (d);
angle2=err2.name;
angle2=angle2(indexdata3);

angle=[angle1 angle2];
fprintf('Ours\n mean median 1 2 5 10 \n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n',mean(angle),median(angle),100*sum(angle<1)/size(angle,2),100*sum(angle<2)/size(angle,2),100*sum(angle<5)/size(angle,2),100*sum(angle<10)/size(angle,2))