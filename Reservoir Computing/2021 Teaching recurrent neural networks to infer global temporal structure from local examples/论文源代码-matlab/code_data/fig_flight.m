%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(6); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 7.0];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .3 .2];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 3.00  4.00 4.00];...     % Subplot position in cm [x,y,w,h]
        [ 0.50 0.00  3.00 3.00];...
        [ 5.00 0.00 14.00 7.00]];

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
CP = [[220 160 100];...
      [220 180 100];...
      [220 200 100]]/255;


%% Parameters
delT = 0.001;                       % Simulation Time-Step
t_waste = 20;                       % Time to settle reservoir transient
t_train = 200;                      % Post-transient time for training
n_w = t_waste/delT;                 % Number of transient samples
n_t = t_train/delT;                 % Number of training samples
n = n_w + n_t;                      % Total number of samples per transform
ind_t = [1:n_t] + n_w;              % Index of training samples
t_ind = [ind_t,...
         ind_t+n,...
         ind_t+2*n,...
         ind_t+3*n,...
         ind_t+4*n,...
         ind_t+5*n,...
         ind_t+6*n,...
         ind_t+7*n,...
         ind_t+8*n,...
         ind_t+9*n];                % Index across 4 example translations


%% Initialize reservoir and Lorenz constant parameters
N = 450;                            % Number of reservoir states
M = 3;                              % Number of Lorenz states
gam = 100;                          % Reservoir responsiveness
sig = 0.004;                        % Attractor influence
c = [.002 .002];                    % Control Parameter
p = 0.1;                            % Reservoir initial density
% Equilibrium point
x0 = zeros(M,1);
c0 = zeros(length(c),1);


%% Initial reservoir and Lorenz random parameters
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);   % Stabilize base matrix
% Input matrices
B = 2*(rand(N,3)-.5)*sig;
C = 2*(rand(N,length(c))-.5).*c;
% Fixed point
r0 = (rand(N,1)*.2+.8) .* sign(rand(N,1)-0.5); % Distribution of offsets

% Lorenz initial condition
Lx0 = rand(3,1)*10;


%% Create reservoir and Lorenz object
% Load example variables A, B, C, r0, and Lx0 that worked when tested
load fig_flight_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system


%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series


%% Train Reservoir
C1 = ones(1,n,4);
C2 = ones(1,n,4);
Cin = [0*C1 1*C1 2*C1 3*C1 0*C1 0*C1 0*C1 1*C1 2*C1 3*C1;...
       0*C2 0*C2 0*C2 0*C2 1*C2 2*C2 3*C2 1*C2 2*C2 3*C2];

% Shift attractor representations
a1 = [1 0 0]';
a2 = [0 0 1]';
X1 = X0 + a1;
X2 = X0 + 2*a1;
X3 = X0 + 3*a1;
X4 = X0 + a2;
X5 = X0 + 2*a2;
X6 = X0 + 3*a2;
X7 = X0 + 1*a1 + 1*a2;
X8 = X0 + 2*a1 + 2*a2;
X9 = X0 + 3*a1 + 3*a2;
X = [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9];
clear X0 X1 X2 X3 X4 X5 X6 X7 X8 X9;

% Drive reservoir
disp('Simulating Reservoir');
RT = R2.train(X,Cin);
RT = RT(:,t_ind);
% Train outputs
disp('Training Wout');
W = lsqminnorm(RT', X(:,t_ind,1)')';           % Use least squares norm
disp(['Training error: ' num2str(norm(W*RT - X(:,t_ind,1)))]);


%% Predict
% Reset reservoir initial conditions
R2.r = RT(:,n_t);

nR = 35000;
nT = 80000;

% Prepare control inputs
nVC1 = 40;
nVC2 = 30;
o = ones(1,nR);

% Flight trajectory
cInds1 = [linspace(0,-nVC1,nT)        o*-nVC1,...
          linspace(-nVC1,-nVC1/2,nT)  o*-nVC1/2,...
          linspace(-nVC1/2,0,nT)      o*0,...
          linspace(0,nVC1/2,nT)       o*nVC1/2,...
          linspace(nVC1/2,nVC1,nT)    o*nVC1];
cInds2 = sin([linspace(0,-2,nT)       o*-2,...
          linspace(-2,-1,nT)          o*-1,...
          linspace(-1,0,nT)           o*0,...
          linspace(0,1,nT)            o*1,...
          linspace(1,2,nT)            o*2]*pi/2);
cInds2 = sqrt(abs(cInds2)).*sign(cInds2)*nVC2;
cInds1 = [cInds1;cInds2];
cDiff1a = [diff(cInds1,1,2), [0;0]];
cInds1a = cat(3,cInds1, cInds1+cDiff1a/2, cInds1+cDiff1a/2, cInds1+cDiff1a);

% Predict
disp('Generating Reservoir Prediction');
% Predict Full Reservoir
RCont = R2.predict_x(cInds1a,W);
% RConta = RCont(:,nR+1:end);
cInds1 = cInds1(:,nR+1:end);
cDiff1 = cDiff1a(:,nR+1:end);
XC = W*RCont(:,nR+1:end);


%% Reservoir
pInd = 1;
subplot('Position',subpN(pInd,:)); cla;
% sizes
sR = 64;
sN = 6;
colormap winter;

% Spline parameters
py1 = .71; py2 = .5; py3 = .29; py4 = 0.08; py5 = -.13;
ps1 = [.21 .22; py1 py1];
ps2 = [.13 .14; py2 py2];
ps3 = [.14 .15; py3 py3];
ps4 = [0 .15; py4 py4];
ps5 = [0 .15; py5 py5];
pse1 = [.22 .343; py1 py1];
pse2 = [.22 .252 .21 .23; py1 .7 .62 .6];
pse3 = [.14 .155 .18 .20; py2 .505 .545 .55];
pse4 = [.14 .155 .16 .18; py2 .493 .41 .4];
pse5 = [.15 .162 .17 .19; py3 .295 .35 .36];
pse6 = [.15 .186 .23 .275; py3 .28 .215 .20];
pse7 = [.15 .162 .22 .30; .08 .09 .16 .18];
pse8 = [.15 .3 .42 .45; .08 .08 .09 .112];
pse9 = [.15 .28 .3 .35; [0 .03]+py5 .10 .14];
pse10 = [.15 .4 .48 .5;  [0 .03]+py5 .01 .112];
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
      
hold on;
% arrows: start
plot_spline(ps1, 'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline(ps2, 'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline(ps3, 'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline(ps4, 'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline(ps5, 'head',1,'headpos',1,'headwidth',4,'headlength',4);
% arrows: extra start
plot_spline(pse1);
plot_spline(pse2);
plot_spline(pse3);
plot_spline(pse4);
plot_spline(pse5);
plot_spline(pse6);
plot_spline(pse7);
plot_spline(pse8);
plot_spline(pse9);
plot_spline(pse10);
% arrows: end
plot_spline([1-pse1(1) .96; py1 py1],'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline([1-pse3(1) .96; py2 py2],'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline([1-pse5(1) .96; py3 py3],'head',1,'headpos',1,'headwidth',4,'headlength',4);
plot_spline([.94 1; py1 py1]);
plot_spline([.94 1; py2 py2]);
plot_spline([.94 1; py3 py3]);
% arrows: extra end
plot_spline(fliplr([1-pse1(1,:);pse1(2,:)]));
plot_spline(fliplr([1-pse2(1,:);pse2(2,:)]));
plot_spline(fliplr([1-pse3(1,:);pse3(2,:)]));
plot_spline(fliplr([1-pse4(1,:);pse4(2,:)]));
plot_spline(fliplr([1-pse5(1,:);pse5(2,:)]));
plot_spline(fliplr([1-pse6(1,:);pse6(2,:)]));
% Loop
plot_spline(psl1,'prntslp',0);
plot_spline(psl2,'prntslp',0);
plot_spline(psl3,'prntslp',0);
% nodes
nP = 60;
cNr = sN/(2*72)*2.54/subp(pInd,4);
cNx = -cNr*cosd(linspace(0,360,nP));
cNy = cNr*sind(linspace(0,360,nP));
patch(cNx+1,cNy+py1,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+py2,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+py3,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+.02,cNy+py4,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+.02,cNy+py5,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
% Reservoir Edges
for i = 1:size(ELis,1)
    plot_spline(EPos(:,:,i), 'head',1,'headpos',.75,'headwidth',4,'headlength',4);
end
% Reservoir Nodes
for i = 1:size(NPos,1)
    plot(NPos(i,1),NPos(i,2),'ko','linewidth',.7,'markersize',sN,'markerfacecolor',CP(2,:));
end
% Reservoir
plot(.5,.43,'ko','linewidth',1,'markersize',sR);
hold off;

% axis
set(gca,'visible',0,'xtick',[],'ytick',[]);
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*-.15];
axis(ax);

% Text
text(1.05,.86,'$\hat{x}_1$',NVTextR{:});
text(1.05,.65,'$\hat{x}_2$',NVTextR{:});
text(1.05,.44,'$\hat{x}_3$',NVTextR{:});
text(-.11,.24,'$c_1$',NVTextR{:});
text(-.11,.03,'$c_2$',NVTextR{:});
drawnow;


%% Colormap
pInd = 2;

pV = unique(cInds1','rows','stable')';

subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
line([-1 -1 1 1 -1]*40, [-1 1 1 -1 -1]*40, 'linewidth',.7,'color','k');
scatter(pV(1,1:100:end),...
        pV(2,1:100:end),...
        10,winter(ceil(size(pV,2)/100)),'filled');
hold off;
axis([-1 1 -1 1]*42);

% Text
text(-.25,-.05,'$-40$',NVTextR{:});
text(1,-.05,'$40$',NVTextR{:});
text(-.14,1,'$40$',NVTextR{:});
text(-.14,.5,'$c_2$',NVTextR{:});
text(.5,-.05,'$c_1$',NVTextR{:});


%% Plot
pInd = 3;
subplot('Position',subpN(pInd,:)); cla;

nF = 500;
cC = winter(nF-2);
cI1 = floor((cInds1(1,:) - min(cInds1(1,:))) / (max(cInds1(1,:))-min(cInds1(1,:))) * (nF-3) + 1);
[XCTDS, dInd] = downsample_curvature(XC,.6,[-10,20]);
fInd = floor(linspace(1,size(XCTDS,2),nF));
hold on;
for i = 1:(nF-2)
    alph = 1 - .9 * ((cDiff1(1,dInd(fInd(i))) ~= 0) | cDiff1(1,dInd(fInd(i+1))) ~= 0);
    plot3(XCTDS(1,fInd(i):fInd(i+1)),...
          XCTDS(2,fInd(i):fInd(i+1)),...
          XCTDS(3,fInd(i):fInd(i+1)),...
          '-','linewidth',.5,'color',[cC(cI1(dInd(fInd(i))),:), alph], 'clipping', 0);
end
axis([-35 40 -30 30 -9 54]*1.2);
view(-10,20);
set(gca,'visible',0);


%% Save
fName = 'fig_flight.pdf';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');