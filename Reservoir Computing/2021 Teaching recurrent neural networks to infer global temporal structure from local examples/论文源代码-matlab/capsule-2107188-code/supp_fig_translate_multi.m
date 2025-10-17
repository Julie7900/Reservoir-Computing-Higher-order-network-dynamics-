%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(9); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                        % Fontsize
fSize = [19 6];                 % Figure Size in cm  [w,h]
fMarg = [0 0 0 0];              % Margins in cm, [l,r,d,u]
subp = [[0 0 6 6];...           % Subplot position in cm [x,y,w,h]
        [6.5 0 6 6];...
        [13 0 6 6]];
    
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
M = 3;                                      % Number of reservoir states
gam = 100;                                  % Reservoir responsiveness
sig = 0.008;                                % Attractor influence
c = .004;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial density
% Equilibrium point
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initial reservoir connectivity and stabilize
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);   % Stabilize base matrix
% Input matrices
B = 2*(rand(N,3)-.5)*sig;
C = 2*(rand(N,1)-.5)*c;
% Fixed point
r0 = (rand(N,1)*.2+.8) .* sign(rand(N,1)-0.5); % Distribution of offsets

% Lorenz initial condition
Lx0 = rand(3,1)*10;


%% Create reservoir object
% Load example variables A, B, C, r0, and Lx0 that worked when tested
load supp_fig_translate_multi_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system


%% a: Data
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series

% Translate time series
a = [1 0 0]';
X1 = X0 + a;
X2 = X0 + 2*a;
X3 = X0 + 3*a;
X = [X0 X1 X2 X3];
clear X1 X2 X3;

b = [0 1 0]';
Y1 = X0 + b;
Y2 = X0 + 2*b;
Y3 = X0 + 3*b;
Y = [X0 Y1 Y2 Y3];
clear Y1 Y2 Y3;

c = [0 0 1]';
Z1 = X0 + c;
Z2 = X0 + 2*c;
Z3 = X0 + 3*c;
Z = [X0 Z1 Z2 Z3];
clear Z1 Z2 Z3;


%% c: Train reservoir
Cin = ones(1,n,4);
Cin = [0*Cin 1*Cin 2*Cin 3*Cin];

% X translation
disp('Simulating Reservoir');
RTX = R2.train(X,Cin);
RTX = RTX(:,t_ind);
disp('Training Woutx');
WoutX = lsqminnorm(RTX', X(:,t_ind,1)')';       % Use least squares norm
disp(norm(WoutX*RTX - X(:,t_ind,1)));
rpRTX = RTX(:,n_t);
clear RTX;


% Y Translation
RTY = R2.train(Y,Cin);
RTY = RTY(:,t_ind);
disp('Training Wouty');
WoutY = lsqminnorm(RTY', Y(:,t_ind,1)')';       % Use least squares norm
disp(norm(WoutY*RTY - Y(:,t_ind,1)));
rpRTY = RTY(:,n_t);
clear RTY;


% Z Translation
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
          -nVS*ones(1,nR+nT),...
          linspace(-nVS,-nVC,nT),...
          -nVC*ones(1,nR),...
          linspace(-nVC,0,nT),...
          -0*ones(1,nR),...
          linspace(0,nVC,nT),...
          nVC*ones(1,nR),...
          linspace(nVC,nVS,nT),...
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
RContX = RContX(:,2*nT+1:end);
RContY = RContY(:,2*nT+1:end);
RContZ = RContZ(:,2*nT+1:end);
cDiff1 = cDiff1a(:,2*nT+1:end);
cInds1 = cInds1(:,2*nT+1:end);
XC = WoutX*RContX;
YC = WoutY*RContY;
ZC = WoutZ*RContZ;


%% Plot
% Views
vw = [-20 20];

nF = 200;
cC = winter(nF-2);
cI = floor((cInds1 - min(cInds1)) / (max(cInds1)-min(cInds1)) * (nF-3) + 1);
[XCTDS, dIndx] = downsample_curvature(XC,1,vw);
[YCTDS, dIndy] = downsample_curvature(YC,1,vw);
[ZCTDS, dIndz] = downsample_curvature(ZC,1,vw);
fIndx = floor(linspace(1,size(XCTDS,2),nF));
fIndy = floor(linspace(1,size(YCTDS,2),nF));
fIndz = floor(linspace(1,size(ZCTDS,2),nF));
ax = [-1 1 -1 1 -1 1]*45 + [0 0 0 0 1 1]*25;

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
xV = -40; yV = -40; zV = -8; aL = 25;
quiver3(xV,yV,zV,aL,0,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,aL,0,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
quiver3(xV,yV,zV,0,0,aL,'linewidth',1,'color','k','autoscale',0,'maxheadsize',0.2,'clipping',0);
hold off;
text(-20,-45,-10,'$x_1$');
text(-45,-15,-10,'$x_2$');
text(-29,-15,5,'$x_3$');
axis(ax);
view(vw(1),vw(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)-.2,'\textbf{a}~~translate $x_1$',NVTitle{:});


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
view(vw(1),vw(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)-.2,'\textbf{b}~~translate $x_2$',NVTitle{:});
% Text
text(-.23,.06,'-40',NVTextR{:});
text(1.15,.06,'40',NVTextR{:});
text(0.5,.03,'c',NVTextR{:});

delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','south','linewidth',.01);
cb.Position = [.3 .05 .4 .025];
cb.Ticks = [];


pInd = 3;
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
view(vw(1),vw(2));
set(gca,'visible',0);
text(labX,subp(pInd,4)-.2,'\textbf{c}~~translate $x_3$',NVTitle{:});
colormap('winter');


%% Save
fName = 'supp_fig_translate_multi';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');