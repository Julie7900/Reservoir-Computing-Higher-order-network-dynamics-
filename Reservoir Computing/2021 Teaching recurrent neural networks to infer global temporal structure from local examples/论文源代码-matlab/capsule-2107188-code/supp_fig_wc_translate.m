%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(9); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [14 4.0];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .05 .45];                % Margins in cm, [l,r,d,u]
subp = [[ 0.00 0.00 10.00 4.00];...     % Subplot position in cm [x,y,w,h]
        [10.00 0.00  4.00 4.00]];
    
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
CO = [[150 100 100];...             % Training output
      [170 100 100];...
      [190 100 100]]/255;
CP = [[220 200 100];...             % Reservoir
      [220 180 100];...
      [220 160 100]]/255;
CB = [100 100 255]/255;             % Input edges
CR = [255 100 100]/255;             % Output edges


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


%% Initialize reservoir and Lorenz random parameters
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
load supp_fig_wc_translate_params.mat;
R2 = ReservoirWC(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system


%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series

% Translate time series
a = [1 0 0]';
X1 = X0 + a;
X2 = X0 + 2*a;
X3 = X0 + 3*a;
Xin = [X0 X1 X2 X3];


%% Train reservoir
Cin = ones(1,n,4);
Cin = [0*Cin 1*Cin 2*Cin 3*Cin];
% Drive reservoir
disp('Simulating Reservoir');
RT = R2.train(Xin,Cin);
RT = RT(:,t_ind);
% Train outputs
disp('Training W');
W = lsqminnorm(RT(1:N/2,:)', Xin(:,t_ind,1)')';      % Use least squares norm
W = [W zeros(3,N/2)];
XT = W*RT;                                  % Projected output
disp(['Training error: ' num2str(norm(XT - Xin(:,t_ind,1)))]);


%% Predict and extrapolate
% Reset reservoir initial conditions
R2.r = RT(:,n_t);

nR = 40000;                     % # time steps to stay in place
nT = 40000;                     % # of time steps to move

% Prepare control inputs
nVS = 40;
nVC = 20;
cInds1 = [linspace(0,-nVS,nT),    -nVS*ones(1,nR+nT),...
          linspace(-nVS,-nVC,nT), -nVC*ones(1,nR),...
          linspace(-nVC,nVC,2*nT), nVC*ones(1,nR),...
          linspace(nVC,nVS,nT),    nVS*ones(1,nR)];
cDiff1a = [diff(cInds1,1,2), 0];
cInds1a = reshape([cInds1; cInds1+cDiff1a/2; cInds1+cDiff1a/2; cInds1+cDiff1a]', [1, length(cInds1), 4]);
% Predict
disp('Generating Reservoir Prediction');

% Predict Full Reservoir
RCont = R2.predict_x(cInds1a,W);
RCont = RCont(:,2*nT+1:end);
cDiff1 = cDiff1a(:,2*nT+1:end);
cInds1a = cInds1a(:,2*nT+1:end,1);
XC = W*RCont;


%% a: Plot predicted reservoir trajectory
pInd = 1;
subplot('Position',subpN(pInd,:)); cla;

nF = 200;                   % Number of frames for plotting color gradient
cC = winter(nF-2);
colormap winter;
cI = floor((cInds1a - min(cInds1a)) / (max(cInds1a)-min(cInds1a)) * (nF-3) + 1);
[XCTDS, dInd] = downsample_curvature(XC,.4,[-10,20]);
fI = floor(linspace(1,size(XCTDS,2),nF));
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * (cDiff1(dInd(fI(i))) ~= 0);
    plot3(XCTDS(1,fI(i):fI(i+1)),XCTDS(2,fI(i):fI(i+1)),XCTDS(3,fI(i):fI(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dInd(fI(i))),:), alph], 'clipping', 0);
end
axis([-45 45 -30 30 0 40]);
delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','south','linewidth',.01);
cb.Position = [.09 .09 .55 .025];
cb.Ticks = [];

tPlInd = 1:20000;
X0PDS = downsample_curvature(X0(:,ind_t(tPlInd),1),.4,[-10,20]);
X1PDS = downsample_curvature(X1(:,ind_t(tPlInd),1),.4,[-10,20]);
X2PDS = downsample_curvature(X2(:,ind_t(tPlInd),1),.4,[-10,20]);
X3PDS = downsample_curvature(X3(:,ind_t(tPlInd),1),.4,[-10,20]);
plot3(X0PDS(1,:),X0PDS(2,:),X0PDS(3,:),...
      'color',[CL(2,:) 1], 'linewidth', .2, 'clipping', 0);
plot3(X1PDS(1,:),X1PDS(2,:),X1PDS(3,:),...
      'color',[CL(2,:) 1], 'linewidth', .2, 'clipping', 0);
plot3(X2PDS(1,:),X2PDS(2,:),X2PDS(3,:),...
      'color',[CL(2,:) 1], 'linewidth', .2, 'clipping', 0);
plot3(X3PDS(1,:),X3PDS(2,:),X3PDS(3,:),...
      'color',[CL(2,:) 1], 'linewidth', .2, 'clipping', 0);
hold off;
view(-10,20);
set(gca,'box',1,'visible',0);
set(gca,'xtick',[-20 20],'ytick',[-20 20],'ztick',[0 40]);
set(gca,'xticklabel',[],'yticklabel',[],'zticklabel',[]);

% Text
% text(labX,subp(pInd,4)+.2,'\textbf{}~~~~~~~~~~~~~~~~~~~~~~~~~~~translate representation',NVTitle{:});
text(0.01,.1,'-40',NVTextR{:});
text(0.94,.1,'40',NVTextR{:});
text(0.5,.03,'c',NVTextR{:});


%% Labels
pInd = 2;
subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);

xV = linspace(0,1,120);
o = ones(1,length(xV));
hold on;
scatter(xV,o*.63,5,repmat(CL(2,:),length(o),1),'filled','s');
scatter(xV,o*.43,5,winter(length(o)),'filled','s');
hold off;

text(.5,.7,'training input',NVTextH{:});
text(.5,.5,'predicted extrapolation',NVTextH{:});

% Axis
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*0];
axis(ax);


%% Save
fName = 'supp_fig_wc_translate';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');