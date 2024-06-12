clc,clear all,close all;
load ./allFaces.mat
% X = faces(:,1:50);
X = faces;

%%
% Build Training and Test sets
nTrain = 30; nTest = 20; nPeople = 30;
Train = zeros(size(X,1),nTrain*nPeople);
Test = zeros(size(X,1),nTest*nPeople);
for k=1:nPeople
    baseind = 0;
    if(k>1) 
        baseind = sum(nfaces(1:k-1));
    end
    inds = baseind + (1:nfaces(k));
    Train(:,(k-1)*nTrain+1:k*nTrain)=X(:,inds(1:nTrain));
    Test(:,(k-1)*nTest+1:k*nTest)=X(:,inds(nTrain+1:nTrain+nTest));
end

%%

M1 = size(Train,2);
Theta1 = zeros(120,M1);

for k=1:M1
    temp = reshape(Train(:,k),n,m);
    tempSmall = imresize(temp,[12 10],'lanczos3');
    Theta1(:,k) = reshape(tempSmall,120,1);
end

for k=1:M1 % Normalize columns of Theta
    Theta1(:,k) = Theta1(:,k)/norm(Theta1(:,k));
end

%%

M2 = size(Test,2);
Theta2 = zeros(120,M2);

for k=1:M2
    temp = reshape(Test(:,k),n,m);
    tempSmall = imresize(temp,[12 10],'lanczos3');
    Theta2(:,k) = reshape(tempSmall,120,1);
end

for k=1:M2 % Normalize columns of Theta
    Theta2(:,k) = Theta2(:,k)/norm(Theta2(:,k));
end

%%
%test data
x1=Test(:,132); %cleanimage

noise_num = 50;
x2=reshape(x1,n,m);
noise = randperm(168);
x2(:,noise(1:noise_num))=0;
x2=reshape(x2,n*m,1);

randvec= randperm(n*m);
first30=randvec(1:floor(.3*length(randvec)));
vals30=uint8(255*rand(size(first30)));
x3=x1;
x3(first30)=vals30; %30%occluded
x4=x1+50*randn(size(x1)); %randomnoise

%%
X =[x1 x2 x3 x4];
Y = zeros(120,4);
for k=1:4
    temp= reshape(X(:,k),n,m);
    tempSmall=imresize(temp,[12 10],'lanczos3');
    Y(:,k)= reshape(tempSmall,120,1);
end

%%
test_num1 = 2;
test_num2 = 3;
test_num3 = 4;
eps =1e-6;

y11=Y(:,test_num1);
cvx_begin;
    variable s11(M1);
    minimize( norm(s11,1));
    subject to
        norm(Theta1*s11 - y11,2) < eps;
cvx_end;

y12=Y(:,test_num2);
cvx_begin;
    variable s12(M1); %sparsevectorofcoefficients
    minimize( norm(s12,1));
    subject to
        norm(Theta1*s12 - y12,2) < eps;
cvx_end;

y13=Y(:,test_num3);
cvx_begin;
    variable s13(M1); %sparsevectorofcoefficients
    minimize( norm(s13,1));
    subject to
        norm(Theta1*s13 - y13,2) < eps;
cvx_end;

%%
y21=Y(:,test_num1);
cvx_begin;
    variable s21(M2); %sparsevectorofcoefficients
    minimize( norm(s21,1));
    subject to
        norm(Theta2*s21 - y21,2) < eps;
cvx_end;

y22=Y(:,test_num2);
cvx_begin;
    variable s22(M2); %sparsevectorofcoefficients
    minimize( norm(s22,1));
    subject to
        norm(Theta2*s22 - y22,2) < eps;
cvx_end;

y23=Y(:,test_num3);
cvx_begin;
    variable s23(M2); %sparsevectorofcoefficients
    minimize( norm(s23,1));
    subject to
        norm(Theta2*s23 - y23,2) < eps;
cvx_end;

%%
x1_pic = mat2gray(reshape(x1,n,m));
x2_pic = mat2gray(reshape(x1,n,m));
x3_pic = mat2gray(reshape(x1,n,m));

figure,
subplot(2,3,1)
plot(s11)
title('s11')

subplot(2,3,2)
plot(s12)
title('s12')

subplot(2,3,3)
plot(s13)
title('s13')

subplot(2,3,4)
plot(s23)
title('s21')

subplot(2,3,5)
plot(s23)
title('s22')

subplot(2,3,6)
plot(s23)
title('s23')

annotation('textbox', [0.1, 0.88, 0.8, 0.1], 'String', 'Train Dictonary', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold')
annotation('textbox', [0.1, 0.43, 0.8, 0.1], 'String', 'Test Dictonary', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold')

figure,
subplot(3,4,5)
imshow(x2_pic)
title('Origin')

subplot(3,4,2)
imshow(mat2gray(reshape(X(:,test_num1),n,m)))
title('Add noise1')

subplot(3,4,3)
x11_re_pic = mat2gray(reshape(Train*(s11./norm(Theta1)'),n,m));
imshow(x11_re_pic)
title('Reconstruction11')

subplot(3,4,6)
imshow(mat2gray(reshape(X(:,test_num2),n,m)))
title('Add noise2')

subplot(3,4,7)
x12_re_pic = mat2gray(reshape(Train*(s12./norm(Theta1)'),n,m));
imshow(x12_re_pic)
title('Reconstruction12')

subplot(3,4,10)
imshow(mat2gray(reshape(X(:,test_num3),n,m)))
title('Add noise3')

subplot(3,4,11)
x13_re_pic = mat2gray(reshape(Train*(s12./norm(Theta1)'),n,m));
imshow(x13_re_pic)
title('Reconstruction13')

subplot(3,4,4)
x21_re_pic = mat2gray(reshape(Test*(s21./norm(Theta2)'),n,m));
imshow(x21_re_pic)
title('Reconstruction21')

subplot(3,4,8)
x22_re_pic = mat2gray(reshape(Test*(s22./norm(Theta2)'),n,m));
imshow(x22_re_pic)
title('Reconstruction22')

subplot(3,4,12)
x23_re_pic = mat2gray(reshape(Test*(s22./norm(Theta2)'),n,m));
imshow(x23_re_pic)
title('Reconstruction23')


[ssimval11,ssimmap11] = ssim(x11_re_pic,x1_pic);
[ssimval12,ssimmap12] = ssim(x12_re_pic,x2_pic);
[ssimval13,ssimmap13] = ssim(x13_re_pic,x3_pic);

[ssimval21,ssimmap21] = ssim(x21_re_pic,x1_pic);
[ssimval22,ssimmap22] = ssim(x22_re_pic,x2_pic);
[ssimval23,ssimmap23] = ssim(x23_re_pic,x3_pic);

%%
figure,
subplot(2,3,1)
imshow(ssimmap11,[]);
title("SSIM Value:" + num2str(ssimval11));

subplot(2,3,2)
imshow(ssimmap12,[]);
title("SSIM Value:" + num2str(ssimval12));

subplot(2,3,3)
imshow(ssimmap13,[]);
title("SSIM Value:" + num2str(ssimval13));

subplot(2,3,4)
imshow(ssimmap21,[]);
title("SSIM Value:" + num2str(ssimval21));

subplot(2,3,5)
imshow(ssimmap22,[]);
title("SSIM Value:" + num2str(ssimval22));

subplot(2,3,6)
imshow(ssimmap23,[]);
title("SSIM Value:" + num2str(ssimval23));

annotation('textbox', [0.1, 0.88, 0.8, 0.1], 'String', 'Reconstruct by Train Dictonary', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold')
annotation('textbox', [0.1, 0.43, 0.8, 0.1], 'String', 'Reconstruct by Test Dictonary', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold')