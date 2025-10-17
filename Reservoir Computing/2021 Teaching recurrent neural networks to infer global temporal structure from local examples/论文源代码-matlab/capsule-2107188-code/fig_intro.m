%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and dimensions
fig = figure(1); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
gr = 0.8;                               % Alpha of gray color
fSize = [19 8.0];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .05 .49];                % Margins in cm, [l,r,d,u]
subp = [[ 0.00 4.20 3.80 3.80];...      % Subplot position in cm [x,y,w,h]
        [ 3.80 4.20 3.80 3.80];...
        [ 7.60 4.20 3.80 3.80];...
        [11.40 4.20 3.80 3.80];...
        [15.20 4.20 3.80 3.80];...
        [ 0.00 0.00 7.50 4.00];...
        [ 7.50 0.00 4.00 4.00];....
        [11.50 0.00 7.50 4.00]];
   
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
CL = [[100 100 150];...                 % Training input
      [100 100 170];...
      [100 100 190]]/255;
CO = [[150 100 100];...                 % Training output
      [170 100 100];...
      [190 100 100]]/255;
CR = [[220 200 100];...                 % Reservoir
      [220 180 100];...
      [220 160 100]]/255;
Cr = [255 100 100]/255;                 % Output edges
CPr = [110 190 240]/255;                % Prediction color
CP = [[220 200 100];...                 % Reservoir
      [220 180 100];...
      [220 160 100]]/255;
CB = [100 100 255]/255;                 % Input edges
CW = [255 100 100]/255;                 % Output edges

% Figure lines
annotation('line',[0.01 .59], [1 1]*.95,'color',[1 1 1]*gr^.5,'linewidth',.3);
annotation('line',[.62 .99], [1 1]*.95,'color',[1 1 1]*gr^.5,'linewidth',.3);


%% Training parameters
delT = 0.001;                       % Simulation Time-Step
t_waste = 20;                       % Time to settle reservoir transient
t_train = 200;                      % Post-transient time for training
n_w = t_waste/delT;                 % Number of transient samples
n_t = t_train/delT;                 % Number of training samples
n = n_w + n_t;                      % Total number of samples per transform
ind_t = (1:n_t) + n_w;              % Index of training samples
t_ind = ind_t;                      % Training index across 2 rotations


%% Initialize reservoir and Lorenz constant parameters
N = 450;                            % Number of reservoir states
M = 3;                              % Number of Lorenz states
gam = 100;                          % Reservoir responsiveness
sig = 0.008;                        % Attractor influence
c = .004;                           % Control Parameter
p = 0.1;                            % Reservoir initial density
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initial reservoir and Lorenz random parameters
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
load fig_intro_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);     % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);                % Lorenz system


%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);               % Generate attractor time series


%% a: Plot training data
pInd = 1; nPlot = 1:20000;
X0p = X0(:,t_ind(nPlot),1);
X0ps1 = downsample_curvature([nPlot;X0p(1,:)],.5);
X0ps2 = downsample_curvature([nPlot;X0p(2,:)],.5);
X0ps3 = downsample_curvature([nPlot;X0p(3,:)-27],.5);

subplot('position',subpN(pInd,:)); cla;
hold on;
plot(X0ps1(1,:),X0ps1(2,:)/75 + 3,'-','color',CL(1,:),'linewidth',.4);
plot(X0ps2(1,:),X0ps2(2,:)/75 + 2.25,'-','color',CL(2,:),'linewidth',.4);
plot(X0ps3(1,:),X0ps3(2,:)/75 + 1.5,'-','color',CL(3,:),'linewidth',.4);
plot_spline([min(nPlot) max(nPlot)*.98; [1 1]*1]);
plot_spline([min(nPlot) max(nPlot); [1 1]*1],'head',1,'linewidth',.3,'headpos',1);
hold off;
axis([min(nPlot) max(nPlot) .5 4]);
set(gca,'visible',0);

% Text
text(labX,subp(pInd,4)-.15,'\textbf{a}~~ Lorenz time series',NVTitle{:});
text(labX,2.35,'$x_1$',NVTitle{:});
text(labX,1.65,'$x_2$',NVTitle{:});
text(labX,0.95,'$x_3$',NVTitle{:});
text(.5,.07,'$t$',NVTextH{:});
drawnow;


%% b: Plot reservoir schematic
% sizes
sR = 59;
sN = 5;

% Spline parameters
ps1 = [0 .22; .71 .71];
ps2 = [0 .14; .50 .50];
ps3 = [0 .15; .29 .29];
pse1 = [.22 .343; .71 .71];
pse2 = [.22 .252 .21 .23; .71 .7 .62 .6];
pse3 = [.14 .155 .18 .20; .50 .505 .545 .55];
pse4 = [.14 .155 .16 .18; .50 .493 .41 .4];
pse5 = [.15 .162 .17 .19; .29 .295 .35 .36];
pse6 = [.15 .186 .23 .275; .29 .28 .215 .20];

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

lw = 0.4;

% Plot
pInd = 2;
subplot('position',subpN(pInd,:)); cla;
hold on;
% arrows: start
plot_spline(ps1, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps2, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps3, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
% arrows: extra start
plot_spline(pse1,'color',CB,'linewidth',lw);
plot_spline(pse2,'color',CB,'linewidth',lw);
plot_spline(pse3,'color',CB,'linewidth',lw);
plot_spline(pse4,'color',CB,'linewidth',lw);
plot_spline(pse5,'color',CB,'linewidth',lw);
plot_spline(pse6,'color',CB,'linewidth',lw);
% arrows: end
plot_spline(fliplr([1-ps1(1,:);ps1(2,:)]),'head',1,'headpos',.5,'color',CW,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(fliplr([1-ps2(1,:);ps2(2,:)]),'head',1,'headpos',.5,'color',CW,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(fliplr([1-ps3(1,:);ps3(2,:)]),'head',1,'headpos',.5,'color',CW,'headwidth',3,'headlength',3,'linewidth',lw);
% arrows: extra end
plot_spline(fliplr([1-pse1(1,:);pse1(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse2(1,:);pse2(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse3(1,:);pse3(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse4(1,:);pse4(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse5(1,:);pse5(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse6(1,:);pse6(2,:)]),'color',CW,'linewidth',lw);
% nodes
plot(0,.71,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(1,:));
plot(0,.50,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(2,:));
plot(0,.29,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(3,:));
plot(1,.71,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(1,:));
plot(1,.50,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(2,:));
plot(1,.29,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(3,:));
% Reservoir Edges
for i = 1:size(ELis,1)
    plot_spline(EPos(:,:,i), 'head',1,'headpos',.75,'headwidth',3,'headlength',3,'linewidth',lw);
end
% Reservoir Nodes
for i = 1:size(NPos,1)
    plot(NPos(i,1),NPos(i,2),'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CR(2,:));
end
% Reservoir
plot(.5,.43,'ko','linewidth',1,'markersize',sR);
hold off;

% axis
set(gca,'visible',0,'xtick',[],'ytick',[]);
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*0 [1 1]*0];
axis(ax);

% Text
text(labX,subp(pInd,4)-.15,'\textbf{b}~~~~~drive reservoir',NVTitle{:});
text(.5,1.1,'train',NVTextH{:});
text(.15,.8,'$B$',NVTextH{:},'color',CB);
text(.85,.8,'$W$',NVTextH{:},'color',CW);
drawnow;


%% Train reservoir
% Drive reservoir
disp('Simulating Reservoir');
RT = R2.train(X0,zeros(1,n,4));
RT = RT(:,t_ind);
% Train outputs
disp('Training W');
W = lsqminnorm(RT', X0(:,t_ind,1)')';
XT = W*RT;
disp(['Training error: ' num2str(norm(XT - X0(:,t_ind,1)))]);


%% c: Plot trained outputs
pInd = 3; 
subplot('position',subpN(pInd,:)); cla;
XTp = XT(:,nPlot);

% Downsample time series for plotting
XResds1 = downsample_curvature([nPlot;XTp(1,:)],.5);
XResds2 = downsample_curvature([nPlot;XTp(2,:)],.5);
XResds3 = downsample_curvature([nPlot;XTp(3,:)-27],.5);

hold on;
plot(XResds1(1,:),XResds1(2,:)/75 + 3,'-','color',CO(1,:),'linewidth',.4);
plot(XResds2(1,:),XResds2(2,:)/75 + 2.25,'-','color',CO(2,:),'linewidth',.4);
plot(XResds3(1,:),XResds3(2,:)/75 + 1.5,'-','color',CO(3,:),'linewidth',.4);
plot_spline([min(nPlot) max(nPlot)*.98; [1 1]*1]);
plot_spline([min(nPlot) max(nPlot); [1 1]*1],'head',1,'linewidth',.3,'headpos',1);
hold off;
axis([min(nPlot) max(nPlot) .5 4]);
set(gca,'visible',0);

% Text
text(labX,subp(pInd,4)-.15,'\textbf{c}~~~~~~~~~~train $W$',NVTitle{:});
text(labX+.03,2.35,'$\hat{x}_1$',NVTitle{:});
text(labX+.03,1.65,'$\hat{x}_2$',NVTitle{:});
text(labX+.03,.95,'$\hat{x}_3$',NVTitle{:});
text(.5,.07,'$t$',NVTextH{:});
drawnow;


%% d: Plot feedback reservoir schematic
% sizes
sR = 59;
sN = 5;

% Spline parameters
py1 = .71; py2 = .5; py3 = .29;
ps1 = [.21 .22; py1 py1];
ps2 = [.13 .14; py2 py2];
ps3 = [.14 .15; py3 py3];
pse1 = [.22 .343; py1 py1];
pse2 = [.22 .252 .21 .23; py1 .7 .62 .6];
pse3 = [.14 .155 .18 .20; py2 .505 .545 .55];
pse4 = [.14 .155 .16 .18; py2 .493 .41 .4];
pse5 = [.15 .162 .17 .19; py3 .295 .35 .36];
pse6 = [.15 .186 .23 .275; py3 .28 .215 .20];
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
pInd = 4; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
% arrows: start
plot_spline(ps1, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps2, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps3, 'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
% arrows: extra start
plot_spline(pse1,'color',CB,'linewidth',lw);
plot_spline(pse2,'color',CB,'linewidth',lw);
plot_spline(pse3,'color',CB,'linewidth',lw);
plot_spline(pse4,'color',CB,'linewidth',lw);
plot_spline(pse5,'color',CB,'linewidth',lw);
plot_spline(pse6,'color',CB,'linewidth',lw);
% arrows: end
plot_spline([1-pse1(1) .96; py1 py1],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([1-pse3(1) .96; py2 py2],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([1-pse5(1) .96; py3 py3],'head',1,'headpos',1,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline([.94 1; py1 py1],'linewidth',lw);
plot_spline([.94 1; py2 py2],'linewidth',lw);
plot_spline([.94 1; py3 py3],'linewidth',lw);
% arrows: extra end
plot_spline(fliplr([1-pse1(1,:);pse1(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse2(1,:);pse2(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse3(1,:);pse3(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse4(1,:);pse4(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse5(1,:);pse5(2,:)]),'color',CW,'linewidth',lw);
plot_spline(fliplr([1-pse6(1,:);pse6(2,:)]),'color',CW,'linewidth',lw);
% Loop
plot_spline(psl1,'linewidth',lw);
plot_spline(psl2,'linewidth',lw);
plot_spline(psl3,'linewidth',lw);
% nodes
plot(1,.71,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CPr);
plot(1,.50,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CPr);
plot(1,.29,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CPr);
% Reservoir Edges
for i = 1:size(ELis,1)
    plot_spline(EPos(:,:,i), 'head',1,'headpos',.75,'headwidth',3,'headlength',3,'linewidth',lw);
end
% Reservoir Nodes
for i = 1:size(NPos,1)
    plot(NPos(i,1),NPos(i,2),'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CR(2,:));
end
% Reservoir
plot(.5,.43,'ko','linewidth',1,'markersize',sR);
hold off;

% axis
set(gca,'visible',0,'xtick',[],'ytick',[]);
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*0];
axis(ax);

% Text
text(labX,subp(pInd,4)-.15,'\textbf{d}~~close feedback loop',NVTitle{:});
text(.82,1.1,'predict',NVTextR{:});
text(1.05,.72,'$x_1''$',NVTextR{:});
text(1.05,.51,'$x_2''$',NVTextR{:});
text(1.05,.30,'$x_3''$',NVTextR{:});
drawnow;


%% Predict time series
R2.r = RT(:,n_t);                           % Initialize reservoir state
RP = R2.predict_x(zeros(1,40000,4),W);      % Evolve feedback reservoir
XP = W*RP;


%% e: Plot reservoir prediction versus Lorenz
pInd = 5;

% Subsample
XPp = XP;
XPps = downsample_curvature(XPp-[0 0 29]',.2,[10,20]);
XTps = downsample_curvature(XTp-[0 0 29]',.2,[10,20]);

subplot('position',subpN(pInd,:)); cla;
plot3(XTps(1,:),XTps(2,:),XTps(3,:),'-','Clipping',0,'linewidth',.2,'color',CL(2,:));
hold on;
plot3(XPps(1,:),XPps(2,:),XPps(3,:),'-','Clipping',0,'linewidth',.2,'color',CPr);
axSh = 20; axL = 40;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh, 'k-','linewidth', .7, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh, 'k-','linewidth', .7, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh, 'k-','linewidth', .7, 'clipping', 0);
axis([-1.3 0.9 -1 1.2 -1 1.2]*20);
view(10,20);
hold off;

% Text
text(labX,subp(pInd,4)-.15,'~~~~~\textbf{e}~~predicted output',NVTitle{:});
text(.52,.84,'---','units','normalized','fontsize',20,'color',CL(2,:));
text(.52,.74,'---','units','normalized','fontsize',20,'color',CPr);
text(.37,.85,'\boldmath$x$:',NVTextR{:});
text(.37,.75,'{\boldmath$x$}$''$:',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% f: Plot Lorenz time series
% Sample and subsample data
nPlot = 1:20000;
X0p = X0(:,ind_t(nPlot),1) - [0 0 27]';
pSh = 10;
X0D = [X0p X0p+[pSh;0;0], X0p+[2*pSh;0;0] X0p+[3*pSh;0;0]];

% Downsample data
nPlotD = 1:4*max(nPlot);
X01Ds = downsample_curvature([nPlotD;X0D(1,:)],.5);
X02Ds = downsample_curvature([nPlotD;X0D(2,:)],.5);
X03Ds = downsample_curvature([nPlotD;X0D(3,:)],.5);
XCDS = [0 0 1 1 2 2 3 3]*.15-.1;
nPlotC = [0 1 1 2 2 3 3 4] * max(nPlot);

% Parameters
pSc = 75;

CN = winter(10);
pInd = 6;
subplot('position',subpN(pInd,:)); cla;
hold on;
plot(X01Ds(1,:),X01Ds(2,:)/pSc + 3,'-','color',CL(1,:),'linewidth',.4);
plot(X02Ds(1,:),X02Ds(2,:)/pSc + 2,'-','color',CL(2,:),'linewidth',.4);
plot(X03Ds(1,:),X03Ds(2,:)/pSc + 1,'-','color',CL(3,:),'linewidth',.4);
for i = 1:3
    plot(nPlotC((-1:0)+2*i),XCDS((-1:0)+2*i),'-','linewidth',.4,'color',CN(i+3,:));
    plot(nPlotC((0:1)+2*i),XCDS((0:1)+2*i),'-','linewidth',.4,'color',CN(i+3,:));
end
plot(nPlotC((7:8)),XCDS((7:8)),'-','linewidth',.4,'color',CN(i+3,:));
plot_spline([min(nPlotD) max(nPlotD)*.99; -.2 -.2]);
plot_spline([min(nPlotD) max(nPlotD); -.2 -.2],'linewidth',.1,'head',1,'headpos',1);

hold off;

% Axis
axis([min(nPlotD) max(nPlotD) [0 4.75]-.6]);
set(gca,'visible',0);

% Text
text(labX,subp(pInd,4)-.1,'\textbf{f}~~~~training input: shifted Lorenz time series',NVTitle{:});
text(labX,2.62,'$x_1$',NVTitle{:});
text(labX,1.9,'$x_2$',NVTitle{:});
text(labX,1.18,'$x_3$',NVTitle{:});
text(labX,0.5,'$c$',NVTitle{:});
text(.5,.02,'$t$',NVTextH{:});
drawnow;


%% g: Plot reservoir schematic
% sizes
sR = 63.5;
sN = 6;

% Spline parameters
ps1 = [0 .22; .71 .71];
ps2 = [0 .14; .50 .50];
ps3 = [0 .15; .29 .29];
ps4 = [0 .15; .08 .08];
pse1 = [.22 .343; .71 .71];
pse2 = [.22 .252 .21 .23; .71 .7 .62 .6];
pse3 = [.14 .155 .18 .20; .50 .505 .545 .55];
pse4 = [.14 .155 .16 .18; .50 .493 .41 .4];
pse5 = [.15 .162 .17 .19; .29 .295 .35 .36];
pse6 = [.15 .186 .23 .275; .29 .28 .215 .20];
pse7 = [.15 .162 .22 .30; .08 .09 .16 .18];
pse8 = [.15 .3 .42 .45; .08 .08 .09 .112];
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
pInd = 7;
subplot('position',subpN(pInd,:)); cla;
hold on;
% arrows: start
plot_spline(ps1, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps2, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(ps3, 'head',1,'headpos',1,'color',CB,'headwidth',3,'headlength',3,'linewidth',lw);
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
plot_spline(fliplr([1-ps1(1,:);ps1(2,:)]),'head',1,'headpos',.5,'color',Cr,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(fliplr([1-ps2(1,:);ps2(2,:)]),'head',1,'headpos',.5,'color',Cr,'headwidth',3,'headlength',3,'linewidth',lw);
plot_spline(fliplr([1-ps3(1,:);ps3(2,:)]),'head',1,'headpos',.5,'color',Cr,'headwidth',3,'headlength',3,'linewidth',lw);
% arrows: extra end
plot_spline(fliplr([1-pse1(1,:);pse1(2,:)]),'color',Cr,'linewidth',lw);
plot_spline(fliplr([1-pse2(1,:);pse2(2,:)]),'color',Cr,'linewidth',lw);
plot_spline(fliplr([1-pse3(1,:);pse3(2,:)]),'color',Cr,'linewidth',lw);
plot_spline(fliplr([1-pse4(1,:);pse4(2,:)]),'color',Cr,'linewidth',lw);
plot_spline(fliplr([1-pse5(1,:);pse5(2,:)]),'color',Cr,'linewidth',lw);
plot_spline(fliplr([1-pse6(1,:);pse6(2,:)]),'color',Cr,'linewidth',lw);
% nodes
plot(0,.71,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(1,:));
plot(0,.50,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(2,:));
plot(0,.29,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CL(3,:));
plot(1,.71,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(1,:));
plot(1,.50,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(2,:));
plot(1,.29,'ko','linewidth',.5,'markersize',sN,'markerfacecolor',CO(3,:));
% Control Node
cNr = sN/(2*72)*2.54/subp(pInd,4);
nP = 120;
cNx = -cNr*cosd(linspace(0,360,nP));
cNy = cNr*sind(linspace(0,360,nP));
patch(cNx,cNy+.08,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.5);
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
colormap('winter');
hold off;

% axis
set(gca,'visible',0,'xtick',[],'ytick',[]);
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*0 [1 1]*-.05];
axis(ax);

% Text
text(labX,subp(pInd,4)-.1,'\textbf{g}\hspace{0.8cm}drive reservoir',NVTitle{:});
drawnow;


%% h: Plot trained outputs
% Sample and subsample data
XTP = XT(:,nPlot,1) - [0 0 27]';
XTD = [XTP XTP+[pSh;0;0], XTP+[2*pSh;0;0] XTP+[3*pSh;0;0]];

% Offset XTD
nPlotD = 1:4*max(nPlot);
XT1Ds = downsample_curvature([nPlotD;XTD(1,:)],.3);
XT2Ds = downsample_curvature([nPlotD;XTD(2,:)],.3);
XT3Ds = downsample_curvature([nPlotD;XTD(3,:)],.3);
XCDS = [0 0 1 1 2 2 3 3]*.2-.1;
nPlotC = [0 1 1 2 2 3 3 4] * max(nPlot);

% Parameters
pSc = 75;
pInd = 8;
subplot('position',subpN(pInd,:)); cla;
hold on;
plot(XT1Ds(1,:),XT1Ds(2,:)/pSc + 3,'-','color',CO(1,:),'linewidth',.4);
plot(XT2Ds(1,:),XT2Ds(2,:)/pSc + 2,'-','color',CO(2,:),'linewidth',.4);
plot(XT3Ds(1,:),XT3Ds(2,:)/pSc + 1,'-','color',CO(3,:),'linewidth',.4);
plot_spline([min(nPlotD) max(nPlotD)*.99; -.2 -.2]);
plot_spline([min(nPlotD) max(nPlotD); -.2 -.2],'linewidth',.1,'head',1,'headpos',1);
hold off;

% Axis
axis([min(nPlotD) max(nPlotD) [0 4.75]-.6]);
set(gca,'visible',0);

% Text
text(labX+.05,subp(pInd,4)-.1,'\textbf{h}~~~~training output: shifted Lorenz time series',NVTitle{:});
text(labX+.05,2.62,'$\hat{x}_1$',NVTitle{:});
text(labX+.05,1.9,'$\hat{x}_2$',NVTitle{:});
text(labX+.05,1.18,'$\hat{x}_3$',NVTitle{:});
text(.5,.02,'$t$',NVTextH{:});
drawnow;


%% Save
fName = 'fig_intro';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');