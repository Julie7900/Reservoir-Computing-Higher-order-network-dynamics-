%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(10); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 5.0];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .1 .4];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 0.00 4.75 4.75];...      % Subplot position in cm [x,y,w,h]
        [ 4.75 0.00 4.75 4.75];...
        [ 9.50 0.00 4.75 4.75];...
        [14.25 0.00 4.75 4.75]];

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

% Figure colors
CL = [[100 100 150];...             % Training input
      [100 100 170];...
      [100 100 190]]/255;


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
N = 600;                                    % Number of reservoir states
M = 3;                                      % Number of Lorenz states
gam = 40;                                  % Reservoir responsiveness
sig = 0.005;                                % Attractor influence
c = .001;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial density
% Equilibrium point
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initial reservoir and Lorenz random parameters
A = zeros(N);
A11 = rand(N/2) .* (rand(N/2) <= p); A11 = A11 - diag(diag(A11));
A21 = rand(N/2) .* (rand(N/2) <= p); A21 = A21 - diag(diag(A11));
A(1:(N/2),1:(N/2)) =                eye(N/2).*rand(N/2,1) + A11;
A(1:(N/2),(1:(N/2))+N/2) =         -eye(N/2).*rand(N/2,1);
A((1:(N/2))+N/2,1:(N/2)) =          eye(N/2).*rand(N/2,1) + A21;
A((1:(N/2))+N/2,(1:(N/2))+N/2) =   -eye(N/2).*rand(N/2,1);
A = sparse(A ./ max(real(eig(A))) * .95);
% Input matrices
B = sig*(rand(N,3)-.5)*2;
C = c*(rand(N,1)-.5)*2;
% Fixed point
r0 = (rand(N,1))*.1 + (rand(N,1)>.5)*(1/1.2-.1);      % All Present

% Lorenz initial condition
Lx0 = rand(3,1)*10;


%% Create reservoir and Lorenz object
% Load example variables A, B, C, r0, and Lx0 that worked when tested
load supp_fig_wc_transform_params.mat;
R2 = ReservoirWC(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system


%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series

% Rotate time series
I = eye(3);
T = zeros(3); T(1,1) = -.012;
X1 = zeros(size(X0));
X2 = zeros(size(X0));
X3 = zeros(size(X0));
for i = 1:4
    X1(:,:,i) = (I+T)*X0(:,:,i);
    X2(:,:,i) = (I+2*T)*X0(:,:,i);
    X3(:,:,i) = (I+3*T)*X0(:,:,i);
end
Xin = [X0 X1 X2 X3];


%% a: Plot Lorenz time series: side view
pInd = 1;
nPlot = 1:30000;
X0DC = downsample_curvature(X0(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X1DC = downsample_curvature(X1(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X2DC = downsample_curvature(X2(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X3DC = downsample_curvature(X3(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);

subplot('position',subpN(pInd,:)); cla;
hold on;
plot3(X0DC(1,:),X0DC(2,:),X0DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
plot3(X1DC(1,:),X1DC(2,:),X1DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
plot3(X2DC(1,:),X2DC(2,:),X2DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
plot3(X3DC(1,:),X3DC(2,:),X3DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
axSh = 22; axL = 34;
plot3([0 axL]-axSh, [0 0]-axSh,   [0 0]-axSh+5,   'k-','linewidth', 1,'clipping',0);
plot3([0 0]-axSh,   [0 axL]-axSh, [0 0]-axSh+5,   'k-','linewidth', 1,'clipping',0);
plot3([0 0]-axSh,   [0 0]-axSh,   [0 axL]-axSh+5, 'k-','linewidth', 1,'clipping',0);
axis([-1.8 0.8 -1.4 1.2 -1.3 1.3]*16);
view(153,15);
hold off;

% Text
text(labX,subp(pInd,4),'\textbf{a}~~~~~~~~~~~~training',NVTitle{:});
text(labX+3.8,subp(pInd,4)+.45,'side view',NVTitle{:});
text(-.065,.23,'$x_1$',NVTextR{:});
text(.84,.15,'$x_2$',NVTextR{:});
text(.58,.9,'$x_3$',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% c: Plot Lorenz time series: top view
pInd = 3;

X0DC2 = downsample_curvature(X0(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X1DC2 = downsample_curvature(X1(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X2DC2 = downsample_curvature(X2(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X3DC2 = downsample_curvature(X3(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);

subplot('position',subpN(pInd,:)); cla;
plot3(X0DC2(1,:),X0DC2(2,:),X0DC2(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
hold on;
plot3(X1DC2(1,:),X1DC2(2,:),X1DC2(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
plot3(X2DC2(1,:),X2DC2(2,:),X2DC2(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
plot3(X3DC2(1,:),X3DC2(2,:),X3DC2(3,:),'k-','Clipping',0,'linewidth',.5,'color',CL(2,:));
axSh = 25; axL = 40;
plot3([0 axL]-axSh, [0 0]-axSh,   [0 0]-axSh+5,   'k-','linewidth', 1,'clipping',0);
plot3([0 0]-axSh,   [0 axL]-axSh, [0 0]-axSh+5,   'k-','linewidth', 1,'clipping',0);
plot3([0 0]-axSh,   [0 0]-axSh,   [0 axL]-axSh+5, 'k-','linewidth', 1,'clipping',0);
plot(-axSh,-axSh,'ko','linewidth',1,'markersize',6);
plot(-axSh,-axSh,'ko','linewidth',2,'markersize',1.4);
axis([-1.2 1.4 -1.25 1.35 -1.4 1.2]*22);
view(0,90);
hold off;

% Text
text(labX,subp(pInd,4),'\textbf{c}~~~~~~~~~~~~training',NVTitle{:});
text(labX+3.8,subp(pInd,4)+.45,'top view',NVTitle{:});
text(.75, .04,'$x_1$',NVTextR{:});
text(-.01,.8,'$x_2$',NVTextR{:});
text(-.09,.04,'$x_3$',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% Train Reservoir
Cin = ones(1,n,4);
Cin = [0*Cin 1*Cin 2*Cin 3*Cin];
% Drive reservoir
disp('Simulating Reservoir');
RT = R2.train(Xin,Cin);
RT = RT(:,t_ind);
% Train outputs
disp('Training W');
W = lsqminnorm(RT(1:N/2,:)', Xin(:,t_ind,1)')';     % Use least squares norm
W = [W zeros(3,N/2)];
disp(['Training error: ' num2str(norm(W*RT - Xin(:,t_ind,1)))]);


%% Predict and extrapolate
% Reset reservoir initial conditions
R2.r = RT(:,n_t);

nR = 40000;                     % # time steps to stay in place
nT = 300000;                    % # of time steps to move

% Prepare control inputs
nVS = 40;
cInds1 = [linspace(0,-nVS,nT), -nVS*ones(1,nR),...
          linspace(-nVS,nVS,nT), nVS*ones(1,nR)];
cDiff1 = [diff(cInds1,1,2), 0];
cInds1a = reshape([cInds1; cInds1+cDiff1/2; cInds1+cDiff1/2; cInds1+cDiff1]', [1, length(cInds1), 4]);
% Predict
disp('Generating Reservoir Prediction');

% Predict Full Reservoir
RCont = R2.predict_x(cInds1a,W);
RCont = RCont(:,1*nT+1:end);
cDiff1 = cDiff1(:,1*nT+1:end);
cInds1 = cInds1(:,1*nT+1:end);
XC = W*RCont;


%% b: Plot predicted reservoir rajectory: side view
pInd = 2;
subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);

nF = 80;
cC = winter(nF-2);
cI = floor((cInds1 - min(cInds1)) / (max(cInds1)-min(cInds1)) * (nF-3) + 1);
[XCTDS, dInd] = downsample_curvature(XC - [0;0;27],.5,[153 15]);
fI = floor(linspace(1,size(XCTDS,2),nF));
hold on;
for i = 1:(nF-2)
    alph = 1 - 0 * (cDiff1(dInd(fI(i))) ~= 0);
    plot3(XCTDS(1,fI(i):fI(i+1)),XCTDS(2,fI(i):fI(i+1)),XCTDS(3,fI(i):fI(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dInd(fI(i))),:), alph], 'clipping', 0);
end
axSh = 22; axL = 34;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
axis([-1.8 0.8 -1.4 1.2 -1.3 1.3]*16);
view(153,15);
hold off;
colormap('winter');
set(gca,'box',1,'visible',0);
set(gca,'xtick',[-20 20],'ytick',[-20 20],'ztick',[0 40]);
set(gca,'xticklabel',[],'yticklabel',[],'zticklabel',[]);

% Colorbar
delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','east','linewidth',.01);
cb.Position = [.45 .3 .01 .35];
cb.Ticks = [];

% Text
text(labX,subp(pInd,4),'\textbf{b}~~~~~~~~~~~prediction',NVTitle{:});
text(.73,.53,'c',NVTextR{:});
text(.87,.27,'$-40$',NVTextR{:},'horizontalalignment','right');
text(.87,.78,'$40$',NVTextR{:},'horizontalalignment','right');


%% d: Plot predicted reservoir rajectory: top view
pInd = 4;

[XCTDS2, dInd] = downsample_curvature(XC - [0;0;27],.5,[0 90]);
fI = floor(linspace(1,size(XCTDS2,2),nF));

subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:(nF-2)
    alph = 1 - 0*(cDiff1(dInd(fI(i))) ~= 0);
    plot3(XCTDS2(1,fI(i):fI(i+1)),XCTDS2(2,fI(i):fI(i+1)),XCTDS2(3,fI(i):fI(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dInd(fI(i))),:), alph], 'clipping', 0);
end
axis([-1.4 1.2 -1.25 1.35 -1.4 1.2]*20);
view(0,90);

axSh = 25; axL = 40;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+5, 'k-','linewidth', 1, 'clipping', 0);
plot(-axSh,-axSh,'ko','linewidth',1,'markersize',6);
plot(-axSh,-axSh,'ko','linewidth',2,'markersize',1.4);
axis([-1.2 1.4 -1.25 1.35 -1.4 1.2]*22);
view(0,90);
hold off;

% Text
text(labX,subp(pInd,4),'\textbf{d}~~~~~~~~~~~prediction',NVTitle{:});


%% Save
fName = 'supp_fig_wc_transform';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');