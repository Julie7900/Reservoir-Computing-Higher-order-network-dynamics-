%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(11); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 4.75];                      % Figure Size in cm  [w,h]
fMarg = [.4 .1 .1 .4];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 0.00 4.75  4.75];...     % Subplot position in cm [x,y,w,h]
        [ 4.75 0.00 4.75  4.75];...
        [ 9.50 0.00 4.75  4.75];...
        [14.25 0.00 4.75  4.75]];

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


%% Lorenz time series
disp('Simulating Attractor');

% Fixed points of Lorenz
fpA = @(rho) [ sqrt(8/3*(rho-1)); sqrt(8/3*(rho-1));rho-1];
fpB = @(rho) [-sqrt(8/3*(rho-1));-sqrt(8/3*(rho-1));rho-1];
% Lorenz initial conditions near eac hfixed point
xsh = [0.6 1.1 0]';

nPLa = linspace(23,24,2);
nPLb = linspace(21,24,4);
H = Lorenz(zeros(3,1), delT, [10 28 8/3]);

% First set of simulations: 2 examples at each fixed point
H.parms = [10 nPLa(1) 8/3]; H.x = fpA(nPLa(1))+4.0*xsh; X0a = H.propagate(n);
H.parms = [10 nPLa(2) 8/3]; H.x = fpA(nPLa(2))+2.8*xsh; X1a = H.propagate(n);
H.parms = [10 nPLa(1) 8/3]; H.x = fpB(nPLa(1))-4.0*xsh; X2a = H.propagate(n);
H.parms = [10 nPLa(2) 8/3]; H.x = fpB(nPLa(2))-2.8*xsh; X3a = H.propagate(n);
Xa = [X0a X1a X2a X3a];


%% a: Plot Lorenz stable fixed point example time series
pInd = 1;
vw = [100 15];
X0DC = downsample_curvature(X0a(:,ind_t,1),.3,vw);
X1DC = downsample_curvature(X1a(:,ind_t,1),.3,vw);
X2DC = downsample_curvature(X2a(:,ind_t,1),.3,vw);
X3DC = downsample_curvature(X3a(:,ind_t,1),.3,vw);

subplot('position',subpN(pInd,:)); cla;
J = CL; J = J(2:end,:);
hold on;
plot3(X0DC(1,:),X0DC(2,:),X0DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',J(3,:));
plot3(X1DC(1,:),X1DC(2,:),X1DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',J(4,:));
plot3(X2DC(1,:),X2DC(2,:),X2DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',J(3,:));
plot3(X3DC(1,:),X3DC(2,:),X3DC(3,:),'k-','Clipping',0,'linewidth',.5,'color',J(4,:));
axSh = 24; axL = 25;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
FPa2 = [0;0;24]; nSc = 22;
ax = [FPa2-nSc FPa2+nSc]';
axis(ax(:));
view(vw);
hold off;

% Text
text(labX,subp(pInd,4),'\textbf{a}~~~~~~~~~training',NVTitle{:});
text(-.05,.04,'$x_1$',NVTextR{:});
text(.64,.15,'$x_2$',NVTextR{:});
text(.09,.65,'$x_3$',NVTextR{:});
text(.3,.85,'---','Units','Normalized','fontsize',24,'VerticalAlignment','middle','color',J(3,:));
text(.52,.85,'$\rho = 23$',NVTextR{:});
text(.3,.75,'---','Units','Normalized','fontsize',24,'VerticalAlignment','middle','color',J(4,:));
text(.52,.75,'$\rho = 24$',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% Initialize reservoir constant parameters
N = 600;                                    % Number of reservoir states
gam = 40;                                   % Reservoir responsiveness
sig = 0.005;                                % Attractor influence
c = .001;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial densitys
% Equilibrium point
x0 = zeros(size(X0a,1),1);
c0 = zeros(length(c),1);


%% Initial reservoir random parameters
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


%% Create reservoir object
% Load example variables A, B, C, and r0 that worked when tested
load supp_fig_wc_bifurcate_params.mat;
R2 = ReservoirWC(A,B,C,r0,x0,c0,delT,gam);


%% Train reservoir
Cin = ones(1,n,4);
Ca = [0*Cin 1*Cin 0*Cin 1*Cin];

% Drive reservoir
disp('Simulating Reservoir a');
RTa = R2.train(Xa,Ca);
RTa = RTa(:,t_ind);
disp('Training Wa');
Wa = lsqminnorm(RTa(1:N/2,:)', Xa(:,t_ind,1)')';       % Use least squares norm
Wa = [Wa zeros(3,N/2)];
disp(['Training error: ' num2str(norm(Wa*RTa - Xa(:,t_ind,1)))]);
r0RTa = RTa(:,n_t/20);
clear RTa;


%% Predict
nRa = 68000;
nRb = 100000;
nT = 60000;

% Prepare control inputs
cInds1 = [linspace(0,5,nT) 5*ones(1,nRa) linspace(5,0,nT) 0*ones(1,nRb) linspace(0,5,nT) 5*ones(1,nRa)];
cDiff1a = [diff(cInds1,1,2), 0];
cInds1 = reshape([cInds1; cInds1+cDiff1a/2; cInds1+cDiff1a/2; cInds1+cDiff1a]', [1, length(cInds1), 4]);

% Predict
disp('Generating Reservoir Prediction');
% Predict Full Reservoir
R2.r = r0RTa;
RCont = R2.predict_x(cInds1,Wa);

% Map reservoir prediction to output prediction
RConta = RCont(:,1:(nT+nRa));
RContb = RCont(:,(1:(nT+nRb))+(nT+nRa));
RContc = RCont(:,(1:(nT+nRa))+(2*nT+nRa+nRb));
cInds1a = cInds1(:,1:(nT+nRa));
cInds1b = cInds1(:,(1:(nT+nRb))+(nT+nRa));
cInds1c = cInds1(:,(1:(nT+nRa))+(2*nT+nRa+nRb));
XCa = Wa*RConta;
XCb = Wa*RContb;
XCc = Wa*RContc;


%% b,d: Plot outputs of predictions
vw = [100 15];
nF = 200;
cC = winter(1.2*nF);

cIa = floor((cInds1a - min(cInds1a)) / (max(cInds1a)-min(cInds1a)) * (nF-3) + 1);
cIb = floor((cInds1b - min(cInds1b)) / (max(cInds1b)-min(cInds1b)) * (nF-3) + 1);
cIc = floor((cInds1c - min(cInds1c)) / (max(cInds1c)-min(cInds1c)) * (nF-3) + 1);

[XCTDSa, dInda] = downsample_curvature(XCa,0.1,vw);
[XCTDSb, dIndb] = downsample_curvature(XCb,0.1,vw);
[XCTDSc, dIndc] = downsample_curvature(XCc,0.1,vw);

% Front
pInd = 2;
subplot('position',subpN(pInd,:)); cla;
fInd = floor(linspace(1,size(XCTDSa,2),nF));
hold on;
for i = 1:(nF-2)
    plot3(XCTDSa(1,fInd(i):fInd(i+1)),...
          XCTDSa(2,fInd(i):fInd(i+1)),...
          XCTDSa(3,fInd(i):fInd(i+1)),...
          '-','linewidth',.5,'color',cC(cIa(dInda(fInd(i))),:), 'clipping', 0);
end
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
hold off;
axis(ax(:));
view(vw);
colormap(cC(1:nF,:));
delete(findall(gcf,'type','colorbar'))
cb = colorbar('location','east','linewidth',.01);
cb.Position = [.268 .3 .01 .35];
cb.Ticks = [];
set(gca,'visible',0);
% Text
text(labX,subp(pInd,4),'\textbf{b}~~~~prediction: $c: 0\rightarrow 5$',NVTitle{:});
text(-.05,.5,'c',NVTextH{:});
text(.007,.75,'5',NVTextH{:});
text(.007,.25,'0',NVTextH{:});
drawnow;


% Front
pInd = 3;
subplot('position',subpN(pInd,:)); cla;
fInd = floor(linspace(1,size(XCTDSb,2),nF));
hold on;
for i = 1:(nF-2)
    plot3(XCTDSb(1,fInd(i):fInd(i+1)),...
          XCTDSb(2,fInd(i):fInd(i+1)),...
          XCTDSb(3,fInd(i):fInd(i+1)),...
          '-','linewidth',.5,'color',cC(cIb(dIndb(fInd(i))),:), 'clipping', 0);
end
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
hold off;
axis(ax(:));
view(vw);
set(gca,'visible',0);
% Text
text(labX,subp(pInd,4),'\textbf{c}~~~~prediction: $c: 5\rightarrow 0$',NVTitle{:});
drawnow;


% Front
pInd = 4;
subplot('position',subpN(pInd,:)); cla;
fInd = floor(linspace(1,size(XCTDSc,2),nF));
hold on;
for i = 1:(nF-2)
    plot3(XCTDSc(1,fInd(i):fInd(i+1)),...
          XCTDSc(2,fInd(i):fInd(i+1)),...
          XCTDSc(3,fInd(i):fInd(i+1)),...
          '-','linewidth',.5,'color',cC(cIc(dIndc(fInd(i))),:), 'clipping', 0);
end
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+22, 'k-','linewidth', 1, 'clipping', 0);
hold off;
axis(ax(:));
view(vw);
set(gca,'visible',0);
% Text
text(labX,subp(pInd,4),'\textbf{d}~~~~prediction: $c: 0\rightarrow 5$',NVTitle{:});
drawnow;




%% Save
fName = 'supp_fig_wc_bifurcate.pdf';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');