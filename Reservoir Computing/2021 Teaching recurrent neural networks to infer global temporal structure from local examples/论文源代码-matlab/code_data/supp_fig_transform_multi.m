%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(10); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 6];                         % Figure Size in cm  [w,h]
fMarg = [0 0 0 0];                      % Margins in cm, [l,r,d,u]
subp = [[0.00 1.1 3.8 3.8];...          % Subplot position in cm [x,y,w,h]
        [3.80 1.1 3.8 3.8];...
        [7.60 1.1 3.8 3.8];...
        [11.4 1.1 3.8 3.8];...
        [15.2 1.1 3.8 3.8]];
    
% Adjust position for margins
subp = subp + [fMarg(1) fMarg(3) -sum(fMarg(1:2)) -sum(fMarg(3:4))];
subpN = subp ./ [fSize(1) fSize(2) fSize(1) fSize(2)];
sRat = subp(:,3) ./ subp(:,4);

% Label Position in cm
labX = -fMarg(1);
labY = fMarg(4)-.2;
set(gcf,'renderer','opengl','Position',[fig.Position(1:2) fSize],'Units','centimeters');
set(gcf,'renderer','opengl','Position',[fig.Position(1:2) fSize],'Units','centimeters');

% Name-Value pairs for text placement
NVTitle = {'Units','centimeters','fontsize',FS};
NVTextH = {'Units','Normalized','fontsize',FS,'HorizontalAlignment','center'};
NVTextR = {'Units','Normalized','fontsize',FS};


%% Parameters
delT = 0.001;                       % Simulation Time-Step
t_waste = 20;                       % Time to settle reservoir transient
t_train = 200;                      % Post-transient time for training
n_w = t_waste/delT;                 % Number of transient samples
n_t = t_train/delT;                 % Number of training samples
n = n_w + n_t;                      % Total number of samples per transform
ind_t = (1:n_t) + n_w;              % Index of training samples
t_ind = [ind_t,...
         ind_t+n,...
         ind_t+2*n,...
         ind_t+3*n];                % Index across 4 example translations


%% Initialize reservoir and Lorenz constant parameters
N = 450;                                    % Number of reservoir states
M = 3;                                      % Number of Lorenz states
gam = 100;                                  % Reservoir responsiveness
sig = 0.008;                                 % Attractor influence
c = .004;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial density
% Equilibrium point
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initial reservoir and Lorenz random parameters
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);   % Stabilize base matrix
% Input matrices
B = 2*(rand(N,3)-.5)*sig;
C = 2*(rand(N,1)-.5)*c;
% Fixed point
r0 = (rand(N,1)*.2+.8) .* sign(rand(N,1)-0.5); % Distribution of offsets

% Lorenz initial condition
Lx0 = rand(3,1)*10;


%% Create reservoir and Lorenz object
% Load example variables A, B, C, r0, and Lx0 that worked when tested
load supp_fig_transform_multi_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system


%% a: Data
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series
I = eye(3);

% Stretch: All directions
TX = zeros(3); TX(3,3) = .012;
X1 = zeros(size(X0));
X2 = zeros(size(X0));
X3 = zeros(size(X0));
for i = 1:4
    X1(:,:,i) = (I+TX)*X0(:,:,i);
    X2(:,:,i) = (I+2*TX)*X0(:,:,i);
    X3(:,:,i) = (I+3*TX)*X0(:,:,i);
end
X = [X0 X1 X2 X3];
clear X1 X2 X3;

% Shear: 
TY = zeros(3); TY(2,1) = .012;
Y1 = zeros(size(X0));
Y2 = zeros(size(X0));
Y3 = zeros(size(X0));
for i = 1:4
    Y1(:,:,i) = (I+TY)*X0(:,:,i);
    Y2(:,:,i) = (I+2*TY)*X0(:,:,i);
    Y3(:,:,i) = (I+3*TY)*X0(:,:,i);
end
Y = [X0 Y1 Y2 Y3];
clear Y1 Y2 Y3;

% Squeeze 
TZ = zeros(3); TZ(1,1) = .012; TZ(2,2) = -.012;
Z1 = zeros(size(X0));
Z2 = zeros(size(X0));
Z3 = zeros(size(X0));
for i = 1:4
    Z1(:,:,i) = (I+TZ)*X0(:,:,i);
    Z2(:,:,i) = (I+2*TZ)*X0(:,:,i);
    Z3(:,:,i) = (I+3*TZ)*X0(:,:,i);
end
Z = [X0 Z1 Z2 Z3];
clear Z1 Z2 Z3;


%% c: Train reservoir
Cin = ones(1,n,4);
Cin = [0*Cin 1*Cin 2*Cin 3*Cin];

% Transformation 1
RTX = R2.train(X,Cin);
RTX = RTX(:,t_ind);
disp('Training Woutx');
WoutX = lsqminnorm(RTX', X(:,t_ind,1)')';       % Use least squares norm
disp(norm(WoutX*RTX - X(:,t_ind,1)));
rpRTX = RTX(:,n_t);
clear RTX;


% Transformation 2
RTY = R2.train(Y,Cin);
RTY = RTY(:,t_ind);
disp('Training Wouty');
WoutY = lsqminnorm(RTY', Y(:,t_ind,1)')';       % Use least squares norm
disp(norm(WoutY*RTY - Y(:,t_ind,1)));
rpRTY = RTY(:,n_t);
clear RTY;


% Transformation 3
RTZ = R2.train(Z,Cin);
RTZ = RTZ(:,t_ind);
disp('Training WoutZ');
WoutZ = lsqminnorm(RTZ', Z(:,t_ind,1)')';       % Use least squares norm
disp(norm(WoutZ*RTZ - Z(:,t_ind,1)));
rpRTZ = RTZ(:,n_t);
clear RTZ;


%% Predict and extrapolate
% Reset reservoir initial conditions

nR = 40000;
nT = 40000;

% Prepare control inputs
nVS = 40;
nVC = 20;
cInds1 = [linspace(0,-nVS,nT),...
          -nVS*ones(1,2*nR),...
          linspace(-nVS,nVS,4*nT),...
          nVS*ones(1,nR)];
cDiff1a = [diff(cInds1,1,2), 0];
cInds1a = reshape([cInds1; cInds1+cDiff1a/2; cInds1+cDiff1a/2; cInds1+cDiff1a]', [1, length(cInds1), 4]);
% Predict
disp('Generating Reservoir Prediction');

% Predict Full Reservoir
R2.r = rpRTX;
RContX = R2.predict_x(cInds1a,WoutX);
R2.r = rpRTY;
RContY = R2.predict_x(cInds1a,WoutY);
R2.r = rpRTZ;
RContZ = R2.predict_x(cInds1a,WoutZ);
RContX = RContX(:,nT+nR:end);
RContY = RContY(:,nT+nR:end);
RContZ = RContZ(:,nT+nR:end);
cDiff1 = cDiff1a(:,nT+nR:end);
cInds1 = cInds1(:,nT+nR:end);
XC = WoutX*RContX;
YC = WoutY*RContY;
ZC = WoutZ*RContZ;


%% Plot
% Views
vw1 = [-20 20];
vw2 = [0 90];

nF = 200;
cC = winter(nF-2);
cI = floor((cInds1 - min(cInds1)) / (max(cInds1)-min(cInds1)) * (nF-3) + 1);
[XCTDS, dIndx] = downsample_curvature(XC,1,vw1);
[YCTDS, dIndy] = downsample_curvature(YC,1,vw1);
[YCTDST, dIndyT] = downsample_curvature(YC,1,vw2);
[ZCTDS, dIndz] = downsample_curvature(ZC,1,vw1);
[ZCTDST, dIndzT] = downsample_curvature(ZC,1,vw2);
fIndx = floor(linspace(1,size(XCTDS,2),nF));
fIndy = floor(linspace(1,size(YCTDS,2),nF));
fIndyT = floor(linspace(1,size(YCTDST,2),nF));
fIndz = floor(linspace(1,size(ZCTDS,2),nF));
fIndzT = floor(linspace(1,size(ZCTDST,2),nF));
ax = [-1 1 -1 1 -1 1]*25 + [0 0 0 0 1 1]*28;
xV = -25; yV = -25; zV = 10; aL = 25;

pInd = 1;
subplot('Position',subpN(pInd,:)); cla;
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * ((cDiff1(dIndx(fIndx(i))) ~= 0) | (cDiff1(dIndx(fIndx(i+1))) ~= 0));
    plot3(XCTDS(1,fIndx(i):fIndx(i+1)),...
          XCTDS(2,fIndx(i):fIndx(i+1)),...
          XCTDS(3,fIndx(i):fIndx(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dIndx(fIndx(i))),:), alph], 'clipping', 0);
end
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
text(xV+20,yV-8,zV,'$x_1$');
text(xV-8,yV+20,zV,'$x_2$');
text(xV-7,yV,zV+25,'$x_3$');
axis(ax);
view(vw1(1),vw1(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)+.9,'\textbf{a}~~~~~~~stretch $x_3$',NVTitle{:});


pInd = 2;
subplot('Position',subpN(pInd,:)); cla;
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * ((cDiff1(dIndy(fIndy(i))) ~= 0) | (cDiff1(dIndy(fIndy(i+1))) ~= 0));
    plot3(YCTDS(1,fIndy(i):fIndy(i+1)),...
          YCTDS(2,fIndy(i):fIndy(i+1)),...
          YCTDS(3,fIndy(i):fIndy(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dIndy(fIndy(i))),:), alph], 'clipping', 0);
end
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
axis(ax);
view(vw1(1),vw1(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)+.9,'\textbf{b}~~~~~~~~~~~~~~~~~~~~~shear $x_2$',NVTitle{:});
text(0.2,1,'side view',NVTextR{:});


pInd = 3;
subplot('Position',subpN(pInd,:)); cla;
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * ((cDiff1(dIndyT(fIndyT(i))) ~= 0) | (cDiff1(dIndyT(fIndyT(i+1))) ~= 0));
    plot3(YCTDST(1,fIndyT(i):fIndyT(i+1)),...
          YCTDST(2,fIndyT(i):fIndyT(i+1)),...
          YCTDST(3,fIndyT(i):fIndyT(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dIndyT(fIndyT(i))),:), alph], 'clipping', 0);
end
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
axis(ax);
view(vw2(1),vw2(2));
set(gca,'visible',0);

% Text
text(0.2,1,'top view',NVTextR{:});
text(0.52,0,'$x_1$',NVTextR{:});
text(-.04,0.54,'$x_2$',NVTextR{:});
text(-.65,-.24,'-40',NVTextR{:});
text(1.54,-.24,'40',NVTextR{:});
text(0.5,-.18,'c',NVTextR{:});

delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','south','linewidth',.01);
cb.Position = [.3 .02 .4 .025];
cb.Ticks = [];


pInd = 4;
subplot('Position',subpN(pInd,:)); cla;
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * ((cDiff1(dIndz(fIndz(i))) ~= 0) | (cDiff1(dIndz(fIndz(i+1))) ~= 0));
    plot3(ZCTDS(1,fIndz(i):fIndz(i+1)),...
          ZCTDS(2,fIndz(i):fIndz(i+1)),...
          ZCTDS(3,fIndz(i):fIndz(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dIndz(fIndz(i))),:), alph], 'clipping', 0);
end
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
axis(ax);
view(vw1(1),vw1(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)+.9,'\textbf{c}~~~~~~~~~~~~~~~~~~~~~shear $x_1$ and $x_2$',NVTitle{:});
text(0.2,1,'side view',NVTextR{:});


pInd = 5;
subplot('Position',subpN(pInd,:)); cla;
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * ((cDiff1(dIndzT(fIndzT(i))) ~= 0) | (cDiff1(dIndzT(fIndzT(i+1))) ~= 0));
    plot3(ZCTDST(1,fIndzT(i):fIndzT(i+1)),...
          ZCTDST(2,fIndzT(i):fIndzT(i+1)),...
          ZCTDST(3,fIndzT(i):fIndzT(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dIndzT(fIndzT(i))),:), alph], 'clipping', 0);
end
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
axis(ax);
view(vw2(1),vw2(2));
text(0.2,1,'top view',NVTextR{:});
set(gca,'visible',0);

colormap('winter');


%% Save
fName = 'supp_fig_transform_multi.pdf';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');