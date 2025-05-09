%% 生成全阴影图像 
clear;close all;clc

% mask directory|掩膜路径
% maskdir = 'C:\Users\Administrator\Desktop\train_B\';
% maskdir = 'C:\Users\Administrator\Desktop\train_B\';
maskdir = 'C:\Users\Administrator\Desktop\vpdataset\train\train_B_256\';
MD = dir([maskdir '/*.png']);

trainmaskDir = 'C:\Users\Administrator\Desktop\vpdataset\train\train_B_256_erode4\';
% trainmaskDir = 'C:\Users\Administrator\Desktop\train_B_dil100\';

mkdir(trainmaskDir);

% ISTD dataset image size 480*640
for i=1:size(MD)
    mname = strcat(maskdir,MD(i).name); 
    m=imread(mname);
    
%     m=imdilate(m,ones(5));
    m=imerode(m,ones(4));
    
%     md=imdilate(m,ones(50));
%     m=md-me;
    
    trainmaskname=[trainmaskDir,MD(i).name];
    
    imwrite(double(m),trainmaskname); 
end