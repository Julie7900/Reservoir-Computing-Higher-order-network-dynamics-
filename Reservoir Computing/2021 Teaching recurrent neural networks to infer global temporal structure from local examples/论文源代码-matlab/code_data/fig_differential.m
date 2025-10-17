%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(4); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 9.3];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .3 .2];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 4.75 4.75 4.75];...      % Subplot position in cm [x,y,w,h]
        [ 5.25 4.75 3.75 4.75];...
        [ 9.00 4.75 4.50 4.75];...
        [14.00 4.75 4.50 4.75];...
        [ 0.00 0.00 4.75 4.75];...
        [ 4.75 0.00 4.75 4.75];...
        [10.00 0.00 9.00 4.75]];

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
CO = [[150 100 100];...
      [170 100 100];...
      [190 100 100]]/255;
CP = [[200 180 120];...
      [220 180 100];...
      [230 210 120]]/255;
% Example colors
CL = [179, 102, 255;...
      140, 102, 255;...
      100, 100, 255;...
      102, 140, 255;...
      102, 179, 255;...
      102, 217, 255]/255;


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
load fig_translate_transform_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system
     

%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series


%% a: Plot reservoir schematic
% sizes
sR = 77.4;
sN = 6;

colormap winter;
% Spline parameters
py1 = .71; py2 = .5; py3 = .29;
ps1 = [.21 .22; py1 py1];
ps2 = [.13 .14; py2 py2];
ps3 = [.14 .15; py3 py3];
ps4 = [0 .15; .08 .08];
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
plot_spline(ps1, 'head',1,'headpos',1);
plot_spline(ps2, 'head',1,'headpos',1);
plot_spline(ps3, 'head',1,'headpos',1);
plot_spline(ps4, 'head',1,'headpos',1);
% arrows: extra start
plot_spline(pse1);
plot_spline(pse2);
plot_spline(pse3);
plot_spline(pse4);
plot_spline(pse5);
plot_spline(pse6);
plot_spline(pse7);
plot_spline(pse8);
% arrows: end
plot_spline([1-pse1(1) .96; py1 py1],'head',1,'headpos',1);
plot_spline([1-pse3(1) .96; py2 py2],'head',1,'headpos',1);
plot_spline([1-pse5(1) .96; py3 py3],'head',1,'headpos',1);
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
patch(cNx+1,cNy+.71,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+.50,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx+1,cNy+.29,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
patch(cNx,cNy+.08,[linspace(0,360,nP/2) linspace(360,0,nP/2)],'clipping',0,'linewidth',.7);
% Reservoir Edges
for i = 1:size(ELis,1)
    plot_spline(EPos(:,:,i), 'head',1,'headpos',.75);
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
ax = [0 sRat(pInd) 0 1]*1 + [[1 1]*.0 [1 1]*-.0];
axis(ax);

% Text
text(labX,subp(pInd,4)-.2,'\textbf{a}~~~~~~~~feedback reservoir',NVTitle{:});
drawnow;


%% Labels
pInd = 2;
subplot('Position',subpN(pInd,:)); cla;
set(gca,'visible',0);

CW = winter(9);

hold on;
plot([0 .21]+.3, [1 1]*.8, '-', 'color', CP(1,:), 'linewidth', 1);
plot([0 .21]+.3, [1 1]*.7, '-', 'color', CP(3,:), 'linewidth', 1);
plot([0 .21]+.3, [1 1]*.4, '-', 'color', CW(5,:), 'linewidth', 1);
plot([0 .21]+.3, [1 1]*.3, '-', 'color', CW(8,:), 'linewidth', 1);
hold off;

text(0,.9,'reservoir state',NVTextR{:});
text(0,.8,'{\boldmath$r$}$''_{c=0}$',NVTextR{:});
text(0,.7,'{\boldmath$\hat{r}$}$''_{c=\Delta c}$',NVTextR{:});
text(0,.5,'output state',NVTextR{:});
text(0,.4,'$W${\boldmath$r$}$''_{c=0}$',NVTextR{:});
text(0,.3,'$W${\boldmath$\hat{r}$}$''_{c=\Delta c}$',NVTextR{:});

% Axis
ax = [0 sRat(pInd) 0 1]*1;
axis(ax);


%% Translation
Cin = ones(1,n,4);
CT = [0*Cin 1*Cin 2*Cin 3*Cin];

a = [1 0 0]';
XT = [X0 X0+a X0+2*a X0+3*a];

% Drive reservoir
disp('Simulating Reservoir');
RT = R2.train(XT,CT);
RT = RT(:,t_ind);
rt0 = RT(:,n_t);
% Train outputs
disp('Training WT');
WT = lsqminnorm(RT', XT(:,t_ind,1)')';        % Use least squares norm
disp(['Training error: ' num2str(norm(WT*RT - XT(:,t_ind,1)))]);
% clear RT;


%% Prediction
R2.r = rt0;
RTP = R2.predict_x(zeros(1,30000,4),WT);


%% Reservoir states and predicted changes: Translation
nPT = 1:10:15000; lnPT = length(nPT);
delRT = zeros(N,lnPT);

MT = A + B*WT;
IN = eye(N);

disp('Computing predicted change in states: translation');
RTS = RTP(:,nPT);
dRTS = gam*(-RTS + tanh(MT*RTS + R2.d));

fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 1:lnPT
    if(i > nInd*lnPT); fprintf('='); nInd = nInd + .01; end
    TD = tanh(MT*RTS(:,i) + R2.d);
    K = (1-TD.^2);
    Kdot = -2*TD.*K.*(MT*dRTS(:,i));

    % Prepare block matrix elements
    SAI = (IN/gam)^-1;
    SB =  IN-K.*A;
    SC =  IN-K.*A;
    SD =  -Kdot.*A;
    
    % Solve for change in reservoir state w.r.t. change in c
    S = -(SC*SAI*SB - SD)\[-SC*SAI IN];
    U = [K.*(C+B*a); Kdot.*(C+B*a)];
    
    dRP = 20*S*U;
    delRT(:,i) = dRP(1:N);
end
fprintf('\n');


%% b: Plot Translation
% subsample
[~,nSrtT] = sort(sum((delRT - mean(delRT,2)).^2,2));
nIndT = (N-3:N);
nSh = ([1 2.2 3.4 4.6])/10;

% Plot
pInd = 3; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndT)
    nP = nSrtT(nIndT(i));
    RTsi = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))],.002);
    RTsi2 = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))+delRT(nP,:)],.002);
    plot(RTsi(1,:),RTsi(2,:)*.25+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RTsi2(1,:),RTsi2(2,:)*.25+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);

% Text
text(labX,subp(pInd,4)-.2,'\textbf{b}\hspace{1.9cm}reservoir trained on translated input',NVTitle{:});
text(.5,-.025,'time',NVTextH{:});
text(labX,3.3,'$r_1$',NVTitle{:});
text(labX,2.4,'$r_2$',NVTitle{:});
text(labX,1.5,'$r_3$',NVTitle{:});
text(labX,0.6,'$r_4$',NVTitle{:});

% 3D View
XTP = WT*RTS;
XTPT = WT*(RTS + delRT);

pInd = 4; 
subplot('position',subpN(pInd,:)); cla;
XTPds = downsample_curvature(XTP,.0,[155,15]) - [0;0;23];
XTPTds = downsample_curvature(XTPT,.0,[155,15]) - [0;0;23];
hold on;
axSh = 19; axL = 40;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh, 'k-', 'linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh, 'k-', 'linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh, 'k-', 'linewidth', 1, 'clipping', 0);
plot3(XTPds(1,:),XTPds(2,:),XTPds(3,:),'-','color',CW(5,:),'clipping',0);
plot3(XTPTds(1,:),XTPTds(2,:),XTPTds(3,:),'-','color',CW(8,:),'clipping',0);
hold off;
set(gca,'visible',0);
axis([-.8 1.4 -.8 1.4 -.5 1.7]*20);
view(155,15);
text(.08,.01,'$x_1$',NVTextR{:});
text(.86,-.03,'$x_2$',NVTextR{:});
text(.68,.85,'$x_3$',NVTextR{:});
drawnow;


%% Transformation
T = zeros(3); T(1,1) = -.012;
X1R = zeros(size(X0));
X2R = zeros(size(X0));
X3R = zeros(size(X0));
for i = 1:4
    X1R(:,:,i) = (eye(3)+T)*X0(:,:,i);
    X2R(:,:,i) = (eye(3)+2*T)*X0(:,:,i);
    X3R(:,:,i) = (eye(3)+3*T)*X0(:,:,i);
end
XR = [X0 X1R X2R X3R];

% Drive reservoir
disp('Simulating Reservoir');
RR = R2.train(XR,CT);
RR = RR(:,t_ind);
rr0 = RR(:,n_t);
disp('Training WR');
WR = lsqminnorm(RR', XR(:,t_ind,1)')';     % Use least squares norm
disp(['Training error: ' num2str(norm(WR*RR - XR(:,t_ind,1)))]);
clear RR;


%% Prediction
R2.r = rr0;
RRP = R2.predict_x(zeros(1,30000,4),WR);


%% Reservoir states and predicted changes: Transformation
nPT = 1:10:15000; lnPT = length(nPT);
delRR = zeros(N,length(nPT));

MR = A + B*WR;
IN = eye(N);

disp('Computing predicted change in states: transformation');
RRS = RRP(:,nPT);
dRRS = gam*(-RRS + tanh(MR*RRS + R2.d));

fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 1:lnPT
    if(i > nInd*lnPT); fprintf('='); nInd = nInd + .01; end
    TD = tanh(MR*RRS(:,i) + R2.d);
    K = (1-TD.^2);
    Kdot = -2*TD.*K.*(MR*dRRS(:,i));
    
    SVal = B*T*WR;
    WinTWr = SVal*(RRS(:,i)) + C;
    WinTWrdot = SVal*dRRS(:,i);
    
    % Prepare block matrix elements
    SAI = (IN/gam)^-1;
    SB =  IN-K.*A;
    SC =  IN-K.*A;
    SD =  -Kdot.*A;
    
    % Solve for change in reservoir state w.r.t. change in c
    S = -(SC*SAI*SB - SD)\[-SC*SAI IN];
    U = [K.*WinTWr; Kdot.*WinTWr + K.*WinTWrdot];
    dRP = -40*S*U;
    delRR(:,i) = dRP(1:N);
end
fprintf('\n');


%% c: Plot transformation
% subsample
[~,nSrtR] = sort(sum((delRR - mean(delRR,2)).^2,2));
nIndR = (N-3:N);
nSh = [1 2.2 3.4 4.6]/10;

% Plot
pInd = 5;
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndR)
    nP = nSrtR(nIndR(i));
    RRsi = downsample_curvature([nPT;RRS(nP,:)-mean(RRS(nP,:))],.002);
    RRsi2 = downsample_curvature([nPT;RRS(nP,:)-mean(RRS(nP,:))+delRR(nP,:)],.002);
    plot(RRsi(1,:),RRsi(2,:)*.25+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RRsi2(1,:),RRsi2(2,:)*.25+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);

% Text
text(labX,subp(pInd,4)-.2,'\textbf{c}\hspace{1.7cm}reservoir trained on transformed input',NVTitle{:});
text(.5,-.025,'time',NVTextH{:});
text(labX,3.3,'$r_1$',NVTitle{:});
text(labX,2.4,'$r_2$',NVTitle{:});
text(labX,1.5,'$r_3$',NVTitle{:});
text(labX,0.6,'$r_4$',NVTitle{:});

% Projection
XRP = WR*RRS;
XRPR = WR*(RRS + delRR);
nSc = .002;

% 3D
pInd = 6; 
subplot('position',subpN(pInd,:)); cla;
XRPds = downsample_curvature(XRP,.0,[155, 15]) - [0;0;23];
XRPRds = downsample_curvature(XRPR,.0,[155, 15]) - [0;0;23];
hold on;
axSh = 19; axL = 40;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh, 'k-',...
      'linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh, 'k-',...
      'linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh, 'k-',...
      'linewidth', 1, 'clipping', 0);
plot3(XRPds(1,:),XRPds(2,:),XRPds(3,:),'-','color',CW(5,:),'clipping',0);
plot3(XRPRds(1,:),XRPRds(2,:),XRPRds(3,:),'-','color',CW(8,:),'clipping',0);
hold off;
set(gca,'visible',0);
axis([-.8 1.4 -.8 1.4 -.5 1.7]*20);
view(155, 15);
text(.08,.01,'$x_1$',NVTextR{:});
text(.86,-.03,'$x_2$',NVTextR{:});
text(.68,.85,'$x_3$',NVTextR{:});
drawnow;


%% Bifurcation
disp('Simulating Attractor');

% Fixed points of Lorenz
fpA = @(rho) [ sqrt(8/3*(rho-1)); sqrt(8/3*(rho-1));rho-1];
fpB = @(rho) [-sqrt(8/3*(rho-1));-sqrt(8/3*(rho-1));rho-1];
% Lorenz initial conditions near eac hfixed point
xsh = [0.6 1.1 0]';

nPLa = linspace(23,24,2);
H = Lorenz(zeros(3,1), delT, [10 28 8/3]);

% 2 examples at each fixed point
H.parms = [10 nPLa(1) 8/3]; H.x = fpA(nPLa(1))+4.0*xsh; X0a = H.propagate(n);
H.parms = [10 nPLa(2) 8/3]; H.x = fpA(nPLa(2))+2.8*xsh; X1a = H.propagate(n);
H.parms = [10 nPLa(1) 8/3]; H.x = fpB(nPLa(1))-4.0*xsh; X2a = H.propagate(n);
H.parms = [10 nPLa(2) 8/3]; H.x = fpB(nPLa(2))-2.8*xsh; X3a = H.propagate(n);
Xa = [X0a X1a X2a X3a];
Ca = [0*Cin 1*Cin 0*Cin 1*Cin];

% Drive reservoir
load fig_bifurcate_period_double_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
disp('Simulating Reservoir');
RTa = R2.train(Xa,Ca);
RTa = RTa(:,t_ind);
% Train outputs
disp('Training WB');
WB = lsqminnorm(RTa', Xa(:,t_ind,1)')';       % Use least squares norm
disp(norm(WB*RTa - Xa(:,t_ind,1)));


%% Analysis
% Solve for reservoir fixed point at c using Newton-Raphson method
MB = A + B*WB;

% Jacobian
JB = @(rv,cv) -IN + (1-tanh(MB*rv+C*cv+R2.d).^2).*MB;
% Cost
EB = @(rv,cv) (-rv + tanh(MB*rv + C*cv + R2.d))'*(-rv + tanh(MB*rv + C*cv + R2.d));
% Derivative
rdotB = @(rv,cv) -rv + tanh(MB*rv + C*cv + R2.d);

% Sweep over c parameters
nC = 1000;
cL = linspace(0,3,nC);
rBL = zeros(N,nC); rBL(:,1) = RTa(:,n_t);
evM = zeros(2,nC); evM(:,1) = eigs(JB(rBL(:,1),cL(1)),2,'largestreal');
EBL = zeros(1,nC);

fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nC
    if(i > nInd*nC); fprintf('='); nInd = nInd + .01; end
    r0BP = rBL(:,i-1);
    for j = 1:10
        r0BPp = -JB(r0BP,cL(i))\rdotB(r0BP,cL(i));
        r0BP = r0BP + r0BPp;
    end
    rBL(:,i) = r0BP;
    EBL(i) = EB(r0BP,cL(i));
    evM(:,i) = eigs(JB(r0BP,cL(i)),2,'largestreal');
end
fprintf('\n');


%% d: Plot Hopf bifurcation
pInd = 7; 

xl = [-max(abs(real(evM(:)))) max(abs(real(evM(:))))];
yl = [min(min(imag(evM(1:2,:)))) max(max(imag(evM(1:2,:))))];
subplot('position',subpN(pInd,:)); cla;
hold on;
cC = winter(nC);
scatter(real(evM(1,:)),imag(evM(1,:)),2,cC,'filled');
scatter(real(evM(2,:)),imag(evM(2,:)),2,cC,'filled');
line([xl(1) xl(2) xl(2) xl(1) xl(1)]*1.5, [yl(1) yl(1) yl(2) yl(2) yl(1)]*1.5, 'color','k','linewidth',1);
line([0 0], [-1 1]*.01+1.5*yl(1), 'color','k','linewidth',.5);
line([-1 1]*.00003+1.5*xl(1), [0 0],'color','k','linewidth',.5);
plot(real(evM(1,1)), imag(evM(1,1)),'ko','linewidth',2,'markersize',2);
plot(real(evM(2,1)), imag(evM(2,1)),'ko','linewidth',2,'markersize',2);
plot(real(evM(1,floor(nC/3))), imag(evM(1,floor(nC/3))),'ko','linewidth',2,'markersize',2);
plot(real(evM(2,floor(nC/3))), imag(evM(2,floor(nC/3))),'ko','linewidth',2,'markersize',2);
plot(.8*xl(2),1.85*yl(1),'ko','linewidth',2,'markersize',2);
hold off;

colormap(winter);
delete(findall(gcf,'type','colorbar'));
cb = colorbar('location','east','linewidth',.01);
cb.Position = [.95 .145 .01 .235];
cb.Ticks = [];
axis([xl yl]*2.2);

text(labX,subp(pInd,4)-.2,'\hspace{.2cm}\textbf{d}\hspace{.2cm}reservoir trained on pre-bifurcated stable fixed points',NVTitle{:});
text(.5,-.02,'real$(\lambda_{\mathrm{max}})$',NVTextH{:});
text(.5,.08,'$0$',NVTextH{:});
text(.05,.5,'imag$(\lambda_{\mathrm{max}})$',NVTextH{:},'rotation',90);
text(.12,.5,'$0$',NVTextH{:});
text(.95,.5,'$c$',NVTextH{:});
text(.912,.8,'$3$',NVTextH{:});
text(.912,.19,'$0$',NVTextH{:});
text(.77,.08,'training',NVTextH{:});

set(gca,'visible',0);


%% Save
fName = 'fig_differential.pdf';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');