%  “平移” 和 “线性变换” 规律

%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(2); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 9.0];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .05 .45];                % Margins in cm, [l,r,d,u]
subp = [[ 0.00 5.00  4.00 4.00];...     % Subplot position in cm [x,y,w,h]
        [ 5.00 5.00 10.00 4.00];...
        [15.00 5.00  4.00 4.00];...
        [ 0.00 0.00  4.75 4.75];...
        [ 4.75 0.00  4.75 4.75];...
        [ 9.50 0.00  4.75 4.75];...
        [14.25 0.00  4.75 4.75]];
    
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
N = 450;                            % Number of reservoir states
M = 3;                              % Number of Lorenz states
gam = 100;                          % Reservoir responsiveness
sig = 0.008;                        % Attractor influence
c = .004;                           % Control Parameter
p = 0.1;                            % Reservoir initial density
% Equilibrium point
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initialize reservoir and Lorenz random parameters
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);       % Stabilize base matrix
% Input matrices
B = 2*sig*(rand(N,M)-.5);
C = 2*c*(rand(N,1)-.5);
% Fixed point
r0 = (rand(N,1)*.2+.8) .* sign(rand(N,1)-0.5);  % Distribution of offsets

% Lorenz initial condition
Lx0 = rand(3,1)*10;


%% Create reservoir and Lorenz object
% Load example variables A, B, C, r0, and Lx0 that worked when tested
load fig_translate_transform_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);     % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);                % Lorenz system


%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                               % Generate time series

% Translate time series
a = [1 0 0]';
X1Ts = X0 + a;
X2Ts = X0 + 2*a;
X3Ts = X0 + 3*a;
XinTs = [X0 X1Ts X2Ts X3Ts];

% Transform time series
I = eye(3);
T = zeros(3); T(1,1) = -.012;
X1Tf = zeros(size(X0));
X2Tf = zeros(size(X0));
X3Tf = zeros(size(X0));
for i = 1:4
    X1Tf(:,:,i) = (I+T)*X0(:,:,i);
    X2Tf(:,:,i) = (I+2*T)*X0(:,:,i);
    X3Tf(:,:,i) = (I+3*T)*X0(:,:,i);
end
XinTf = [X0 X1Tf X2Tf X3Tf];


%% Train reservoir
Cin = ones(1,n,4);
Cin = [0*Cin 1*Cin 2*Cin 3*Cin];

% Translation
disp('Simulating Reservoir');
RT = R2.train(XinTs,Cin);
RT = RT(:,t_ind);
% Train outputs
disp('Training W');
WTs = lsqminnorm(RT', XinTs(:,t_ind,1)')';      % Use least squares norm
XTs = WTs*RT;                                   % Projected output
disp(['Training error: ' num2str(norm(XTs - XinTs(:,t_ind,1)))]);
rsTs = RT(:,n_t);
clear RT;

% Transformation
disp('Simulating Reservoir');
RT = R2.train(XinTf,Cin);
RT = RT(:,t_ind);
% Train outputs
disp('Training W');
WTf = lsqminnorm(RT', XinTf(:,t_ind,1)')';      % Use least squares norm
XTf = WTf*RT;                                   % Projected output
disp(['Training error: ' num2str(norm(XTf - XinTf(:,t_ind,1)))]);
rsTf = RT(:,n_t);
clear RT;


%% a: Plot feedback reservoir
% sizes
sR = 63.5;
sN = 5;
lw = 0.4;

% Spline parameters
py1 = .71; py2 = .5; py3 = .29; py4 = 0.08;
ps1 = [.21 .22; py1 py1];
ps2 = [.13 .14; py2 py2];
ps3 = [.14 .15; py3 py3];
ps4 = [0 .15; py4 py4];
pse1 = [.22 .343; py1 py1];
pse2 = [.22 .252 .21 .23; py1 .7 .62 .6];
pse3 = [.14 .155 .18 .20; py2 .505 .545 .55];
pse4 = [.14 .155 .16 .18; py2 .493 .41 .4];
pse5 = [.15 .162 .17 .19; py3 .295 .35 .36];
pse6 = [.15 .186 .23 .275; py3 .28 .215 .20];
pse7 = [.15 .162 .22 .30; .08 .09 .16 .18];
pse8 = [.15 .3 .42 .45; .08 .08 .09 .112];
psl1 = [1-ps1(1,2) .80  .78  .3  .12    ps1(1,2);...
               py1 .715 .75  .79  .722    py1];
psl2 = [1-ps2(1,2) .88  .85  .07  .03    ps2(1,2);...
               py2 .51  .74  .74  .531   py2];
psl3 = [1-ps3(1,2) .88  .86  .1   .04    ps3(1,2);...
               py3 .28  .16  .15  .266   py3];
% Node and Edge Positions
NPos = [.4 .62; .3 .4; .7 .5; .5 .2; .68 .3; .45 .4];
ELis = [1 2; 1 3; 3 1; 2 4; 4 5; 5 2; 6 2; 6 5; 4 3; 3 5; 3 6];
EPos = zeros(2,4,size(ELis,1));
for i = 1:size(ELis,1)
    xP = linspace(NPos(ELis(i,1),1),NPos(ELis(i,2),1),13);
    yP = linspace(NPos(ELis(i,1),2),NPos(ELis(i,2),2),13);
    if(abs(diff(NPos(ELis(i,:),1))) > abs(diff(NPos(ELis(i,:),2))))
        EPos(:,:,i) = [xP([1 5 9 13]);yP([1 1 4 end])];
    else
        EPos(:,:,i) = [xP([1 1 4 end]);yP([1 5 9 13])];
    end
end
           
% Plot
pInd = 1; 
subplot('position',subpN(pInd,:)); cla;
hold on;
% arrows: start
plot_spline(ps1, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps2, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps3, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps4, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
% arrows: extra start
plot_spline(pse1,'color',CB,'linewidth',lw);
plot_spline(pse2,'color',CB,'linewidth',lw);
plot_spline(pse3,'color',CB,'linewidth',lw);
plot_spline(pse4,'color',CB,'linewidth',lw);
plot_spline(pse5,'color',CB,'linewidth',lw);
plot_spline(pse6,'color',CB,'linewidth',lw);
plot_spline(pse7,'color',CB,'linewidth',lw);
plot_spline(pse8,'color',CB,'linewidth',lw);
% arrows: end
plot_spline([1-pse1(1) .96; py1 py1],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([1-pse3(1) .96; py2 py2],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([1-pse5(1) .96; py3 py3],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([.94 1; py1 py1],'linewidth',lw);
plot_spline([.94 1; py2 py2],'linewidth',lw);
plot_spline([.94 1; py3 py3],'linewidth',lw);
% arrows: extra end
plot_spline(fliplr([1-pse1(1,:);pse1(2,:)]),'color',CR,'linewidth',lw);
plot_spline(fliplr([1-pse2(1,:);pse2(2,:)]),'color',CR,'linewidth',lw);
plot_spline(fliplr([1-pse3(1,:);pse3(2,:)]),'color',CR,'linewidth',lw);
plot_spline(fliplr([1-pse4(1,:);pse4(2,:)]),'color',CR,'linewidth',lw);
plot_spline(fliplr([1-pse5(1,:);pse5(2,:)]),'color',CR,'linewidth',lw);
plot_spline(fliplr([1-pse6(1,:);pse6(2,:)]),'color',CR,'linewidth',lw);
% Loop
plot_spline(psl1,'prntslp',0,'linewidth',lw);
plot_spline(psl2,'prntslp',0,'linewidth',lw);
plot_spline(psl3,'prntslp',0,'linewidth',lw);
% nodes
nP = 60;
cNr = sN/(2*72)*2.54/subp(pInd,4);
cNx = -cNr*cosd(linspace(0,360,nP));
cNy = cNr*sind(linspace(0,360,nP));
patch(cNx+1,cNy+.71,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+.50,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+.29,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx,cNy+.08,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
% Reservoir Edges
for i = 1:size(ELis,1)
    plot_spline(EPos(:,:,i), 'head',1,'headpos',.75,'headwidth',3,'headlength',3,'linewidth',lw);
end
% Reservoir Nodes
for i = 1:size(NPos,1)
    plot(NPos(i,1),NPos(i,2),'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CP(2,:));
end
% Reservoir
plot(.5,.43,'ko','linewidth',1,'markersize',sR);
hold off;

% axis
set(gca,'visible',0,'xtick',[],'ytick',[]);
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*0];
axis(ax);

% Text
text(labX,subp(pInd,4)+.2,'\textbf{a}~~~close feedback loop',NVTitle{:});
text(1.05,.72,'$x''_1$',NVTextR{:});
text(1.05,.51,'$x''_2$',NVTextR{:});
text(1.05,.30,'$x''_3$',NVTextR{:});
text(-.1,.075,'$c$',NVTextR{:});
drawnow;


%% Predict and extrapolate
% Translate
% Reset reservoir initial conditions
R2.r = rsTs;
nR = 40000;                     % # time steps to stay in place
nT = 40000;                     % # of time steps to move
% Prepare control inputs
nVS = 40;
nVC = 20;
cInds1Ts = [linspace(0,-nVS,nT),    -nVS*ones(1,nR+nT),...
            linspace(-nVS,-nVC,nT), -nVC*ones(1,nR),...
            linspace(-nVC,nVC,2*nT), nVC*ones(1,nR),...
            linspace(nVC,nVS,nT),    nVS*ones(1,nR)];
cDiff1aTs = [diff(cInds1Ts,1,2), 0];
cInds1aTs = reshape([cInds1Ts; cInds1Ts+cDiff1aTs/2; cInds1Ts+cDiff1aTs/2; cInds1Ts+cDiff1aTs]', [1, length(cInds1Ts), 4]);
% Predict
disp('Generating Reservoir Prediction: Translate');
% Predict Full Reservoir
RCont = R2.predict_x(cInds1aTs,WTs);
RCont = RCont(:,2*nT+1:end);
cDiff1Ts = cDiff1aTs(:,2*nT+1:end);
cInds1aTs = cInds1aTs(:,2*nT+1:end,1);
XCTs = WTs*RCont;


% Transform
% Reset reservoir initial conditions
R2.r = rsTf;
nR = 40000;                     % # time steps to stay in place
nT = 300000;                    % # of time steps to move
% Prepare control inputs
nVS = 40;
cInds1Tf = [linspace(0,-nVS,nT), -nVS*ones(1,nR),...
            linspace(-nVS,nVS,nT), nVS*ones(1,nR)];
cDiff1Tf = [diff(cInds1Tf,1,2), 0];
cInds1aTf = reshape([cInds1Tf; cInds1Tf+cDiff1Tf/2; cInds1Tf+cDiff1Tf/2; cInds1Tf+cDiff1Tf]', [1, length(cInds1Tf), 4]);
% Predict
disp('Generating Reservoir Prediction: Transform');
% Predict Full Reservoir
RCont = R2.predict_x(cInds1aTf,WTf);
RCont = RCont(:,1*nT+1:end);
cDiff1Tf = cDiff1Tf(:,1*nT+1:end);
cInds1Tf = cInds1Tf(:,1*nT+1:end);
XCTf = WTf*RCont;


%% b: Plot predicted reservoir trajectory
pInd = 2;
subplot('Position',subpN(pInd,:)); cla;

colormap winter;
nF = 200;                   % Number of frames for plotting color gradient
cC = winter(nF-2);
cI = floor((cInds1aTs - min(cInds1aTs)) / (max(cInds1aTs)-min(cInds1aTs)) * (nF-3) + 1);
[XCTDS, dInd] = downsample_curvature(XCTs,.4,[-10,20]);
fI = floor(linspace(1,size(XCTDS,2),nF));
hold on;
for i = 1:(nF-2)
    alph = 1 - .85 * (cDiff1Ts(dInd(fI(i))) ~= 0);
    plot3(XCTDS(1,fI(i):fI(i+1)),XCTDS(2,fI(i):fI(i+1)),XCTDS(3,fI(i):fI(i+1)),...
          '-','linewidth',.5,'color',[cC(cI(dInd(fI(i))),:), alph], 'clipping', 0);
end
axis([-45 45 -30 30 0 40]);
delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','south','linewidth',.01);
cb.Position = [.32 .59 .43 .025];
cb.Ticks = [];

tPlInd = 1:20000;
X0PDS = downsample_curvature(X0(:,ind_t(tPlInd),1),.4,[-10,20]);
X1PDS = downsample_curvature(X1Ts(:,ind_t(tPlInd),1),.4,[-10,20]);
X2PDS = downsample_curvature(X2Ts(:,ind_t(tPlInd),1),.4,[-10,20]);
X3PDS = downsample_curvature(X3Ts(:,ind_t(tPlInd),1),.4,[-10,20]);
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
text(labX,subp(pInd,4)+.2,'\textbf{b}~~~~~~~~~~~~~~~~~~~~~~~~~~~translate representation',NVTitle{:});
text(0.01,.1,'-40',NVTextR{:});
text(0.94,.1,'40',NVTextR{:});
text(0.5,.03,'c',NVTextR{:});


%% Labels
pInd = 3;
subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);

xV = linspace(0,1,120);
o = ones(1,length(xV));
hold on;
scatter(xV,o*.73,5,repmat(CL(2,:),length(o),1),'filled','s');
scatter(xV,o*.43,5,winter(length(o)),'filled','s');
hold off;

text(.5,.8,'training input',NVTextH{:});
text(.5,.5,'predicted extrapolation',NVTextH{:});

% Axis
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*0];
axis(ax);




%% c: Plot Lorenz time series: side view
pInd = 4;
nPlot = 1:30000;
X0DC = downsample_curvature(X0(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X1DC = downsample_curvature(X1Tf(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X2DC = downsample_curvature(X2Tf(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);
X3DC = downsample_curvature(X3Tf(:,ind_t(nPlot))-[0;0;27],.4,[153 15]);

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
text(labX,subp(pInd,4),'\textbf{c}~~~~~~~~~~~~training',NVTitle{:});
text(labX+3.8,subp(pInd,4)+.45,'side view',NVTitle{:});
text(-.065,.23,'$x_1$',NVTextR{:});
text(.84,.15,'$x_2$',NVTextR{:});
text(.58,.9,'$x_3$',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% e: Plot Lorenz time series: top view
pInd = 6;

X0DC2 = downsample_curvature(X0(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X1DC2 = downsample_curvature(X1Tf(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X2DC2 = downsample_curvature(X2Tf(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);
X3DC2 = downsample_curvature(X3Tf(:,ind_t(nPlot))-[0;0;27],.4,[0 90]);

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
text(labX,subp(pInd,4),'\textbf{e}~~~~~~~~~~~~training',NVTitle{:});
text(labX+3.8,subp(pInd,4)+.45,'top view',NVTitle{:});
text(.75, .04,'$x_1$',NVTextR{:});
text(-.01,.8,'$x_2$',NVTextR{:});
text(-.09,.04,'$x_3$',NVTextR{:});
set(gca,'visible',0);
drawnow;



%% d: Plot predicted reservoir rajectory: side view
pInd = 5;
subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);

nF = 80;
cC = winter(nF-2);
cI = floor((cInds1Tf - min(cInds1Tf)) / (max(cInds1Tf)-min(cInds1Tf)) * (nF-3) + 1);
[XCTDS, dInd] = downsample_curvature(XCTf - [0;0;27],.5,[153 15]);
fI = floor(linspace(1,size(XCTDS,2),nF));
hold on;
for i = 1:(nF-2)
    alph = 1 - 0 * (cDiff1Tf(dInd(fI(i))) ~= 0);
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

% Text
text(labX,subp(pInd,4),'\textbf{d}~~~~~~~~~~~prediction',NVTitle{:});
text(.73,.53,'c',NVTextR{:});
text(.87,.27,'$-40$',NVTextR{:},'horizontalalignment','right');
text(.87,.78,'$40$',NVTextR{:},'horizontalalignment','right');


%% d: Plot predicted reservoir rajectory: top view
pInd = 7;

[XCTDS2, dInd] = downsample_curvature(XCTf - [0;0;27],.5,[0 90]);
fI = floor(linspace(1,size(XCTDS2,2),nF));

subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:(nF-2)
    alph = 1 - 0*(cDiff1Tf(dInd(fI(i))) ~= 0);
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
text(labX,subp(pInd,4),'\textbf{f}~~~~~~~~~~~prediction',NVTitle{:});


%% Save
fName = 'fig_translate_transform';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
% saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');