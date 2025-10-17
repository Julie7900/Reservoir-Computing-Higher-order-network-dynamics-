%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(3); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 10];                        % Figure Size in cm  [w,h]
fMarg = [.4 .1 .3 .2];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 5.25 4.75  4.75];...     % Subplot position in cm [x,y,w,h]
        [ 4.75 5.25 4.75  4.75];...
        [ 9.50 5.25 4.75  4.75];...
        [14.25 5.25 4.75  4.75];...
        [ 0.00 0.00 4.75  4.75];...
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
NVTextHD = {'Units','Data','fontsize',FS,'HorizontalAlignment','center'};
NVTextRD = {'Units','Data','fontsize',FS};

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
lw = 0.3;

pInd = 1;
vw = [100 15];
X0DC = downsample_curvature(X0a(:,ind_t,1),.2,vw);
X1DC = downsample_curvature(X1a(:,ind_t,1),.2,vw);
X2DC = downsample_curvature(X2a(:,ind_t,1),.2,vw);
X3DC = downsample_curvature(X3a(:,ind_t,1),.2,vw);

subplot('position',subpN(pInd,:)); cla;
J = CL; J = J(2:end,:);
hold on;
plot3(X0DC(1,:),X0DC(2,:),X0DC(3,:),'k-','Clipping',0,'linewidth',lw,'color',J(3,:));
plot3(X1DC(1,:),X1DC(2,:),X1DC(3,:),'k-','Clipping',0,'linewidth',lw,'color',J(4,:));
plot3(X2DC(1,:),X2DC(2,:),X2DC(3,:),'k-','Clipping',0,'linewidth',lw,'color',J(3,:));
plot3(X3DC(1,:),X3DC(2,:),X3DC(3,:),'k-','Clipping',0,'linewidth',lw,'color',J(4,:));
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
text(-.03,.04,'$x_1$',NVTextR{:});
text(.62,.16,'$x_2$',NVTextR{:});
text(.09,.65,'$x_3$',NVTextR{:});
text(.3,.85,'---','Units','Normalized','fontsize',24,'VerticalAlignment','middle','color',J(3,:));
text(.52,.85,'$\rho = 23$',NVTextR{:});
text(.3,.75,'---','Units','Normalized','fontsize',24,'VerticalAlignment','middle','color',J(4,:));
text(.52,.75,'$\rho = 24$',NVTextR{:});
set(gca,'visible',0);
drawnow;


%% Initialize reservoir constant parameters
N = 450;                                    % Number of reservoir states
gam = 100;                                  % Reservoir responsiveness
sig = 0.008;                                % Attractor influence
c = .004;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial densitys
% Equilibrium point
x0 = zeros(size(X0a,1),1);
c0 = zeros(length(c),1);


%% Initial reservoir random parameters
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);   % Stabilize base matrix
% Input matrices
B = 2*(rand(N,3)-.5)*sig;
C = 2*(rand(N,1)-.5)*c;
% Fixed point
r0 = (rand(N,1)*.2+.8).* sign(rand(N,1)-0.5); % Distribution of offset


%% Create reservoir object
% Load example variables A, B, C, and r0 that worked when tested
load fig_bifurcate_period_double_params.mat;
R2 = ReservoirTanh(A,B,C,r0,x0,c0,delT,gam);


%% Train reservoir
Cin = ones(1,n,4);
Ca = [0*Cin 1*Cin 0*Cin 1*Cin];


% Drive reservoir
disp('Simulating Reservoir');
RTa = R2.train(Xa,Ca);
RTa = RTa(:,t_ind);

disp('Training Wa');
Wa = lsqminnorm(RTa', Xa(:,t_ind,1)')';       % Use least squares norm
disp(['Training error: ' num2str(norm(Wa*RTa - Xa(:,t_ind,1)))]);


%% Predict
nRa = 90000;
nRb = 150000;
nT = 60000;

% Prepare control inputs
cInds1 = [linspace(0,5,nT) 5*ones(1,nRa) linspace(5,0,nT) 0*ones(1,nRb) linspace(0,5,nT) 5*ones(1,nRa)];
cDiff1a = [diff(cInds1,1,2), 0];
cInds1 = reshape([cInds1; cInds1+cDiff1a/2; cInds1+cDiff1a/2; cInds1+cDiff1a]', [1, length(cInds1), 4]);

% Predict
disp('Generating Reservoir Prediction');
% Predict Full Reservoir
R2.r = RTa(:,n_t/10);
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

figure(3);

% Front
pInd = 2;
subplot('position',subpN(pInd,:)); cla;
fInd = floor(linspace(1,size(XCTDSa,2),nF));
hold on;
for i = 1:(nF-2)
    plot3(XCTDSa(1,fInd(i):fInd(i+1)),...
          XCTDSa(2,fInd(i):fInd(i+1)),...
          XCTDSa(3,fInd(i):fInd(i+1)),...
          '-','linewidth',lw,'color',cC(cIa(dInda(fInd(i))),:), 'clipping', 0);
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
cb.Position = [.268 .68 .01 .18];
cb.Ticks = [];
set(gca,'visible',0);
% Text
text(labX,subp(pInd,4),'\textbf{b}~~~~prediction: $c: 0\rightarrow 5$',NVTitle{:});
text(-.05,.51,'c',NVTextH{:});
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
          '-','linewidth',lw,'color',cC(cIb(dIndb(fInd(i))),:), 'clipping', 0);
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
          '-','linewidth',lw,'color',cC(cIc(dIndc(fInd(i))),:), 'clipping', 0);
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


%% Period Doubling Examples
% Parameters
delT = 0.0002;
rhoV = [100.3 100.2 100.1 100 99.8];
tW = 100;
tT = 6;

X = zeros(3,tT/delT,length(rhoV));

L = Lorenz([0 28 96]',delT,[10 100 8/3]);
for i = 1:length(rhoV)
    L.parms = [10 rhoV(i) 8/3];
    L.propagate(tW/delT);
    XP = L.propagate(tT/delT);
    X(:,:,i) = XP(:,:,1);
end

% Find contiguous subsection to plot
xss1a = [-10.6 -27.96 55.23]';
xss1b = [-8.911 -23.51 53.12]';
xss2a = [-10.67 -28.01 55.29]';
xss2b = [-8.288 -22.04 52.4]';

XSS1 = cell(length(rhoV),1);
XSS2 = cell(2,1);
for i = 1:4
    XPI = find(sum([(X(1:2,:,i)>xss1a(1:2)) ; (X(3,:,i)<xss1a(3))] &...
                   [(X(1:2,:,i)<xss1b(1:2)) ; (X(3,:,i)>xss1b(3))])==3);
    XPII = find(diff(XPI)>1);
    XSS1{i} = X(:,(XPI(XPII(1)+1):XPI(XPII(2))),i);
end

XPI = find(sum([(X(1:2,:,end)>xss2a(1:2)) ; (X(3,:,end)<xss2a(3))] &...
               [(X(1:2,:,end)<xss2b(1:2)) ; (X(3,:,end)>xss2b(3))])==3);
XPII = find(diff(XPI)>1);
XSS2{1} = X(:,(XPI(XPII(1)+1):XPI(XPII(2))),end);
XSS2{2} = X(:,(XPI(XPII(2)+1):XPI(XPII(3))),end);

% Solve for poincarre section of examples
XPE = cell(size(X,3),1);
for i = 1:size(X,3)
    Xp = X(:,:,i);
    x0I = find(diff(Xp(1,:)>0));
    dx0I = Xp(1,x0I+1) - Xp(1,x0I);
    x0I = x0I(dx0I>0);
    xF = -Xp(1,x0I)./(Xp(1,x0I+1)-Xp(1,x0I));
    XPE{i} = Xp(:,x0I) + (Xp(:,x0I+1)-Xp(:,x0I)).*xF;
    disp(i);
end


%% e,f: Plot period doubling examples
vw = [60 15];
X0DC = downsample_curvature(X(:,:,1),2,vw);
X1DC = downsample_curvature(X(:,:,2),2,vw);
X2DC = downsample_curvature(X(:,:,3),2,vw);
X3DC = downsample_curvature(X(:,:,4),2,vw);
X4DC = downsample_curvature(X(:,:,5),2,vw);

% Scale and shift subsection
Sc = 10;
Sh = [110;210;-500];
% ssSh = [0;0;0]'

pInd = 5;
subplot('position',subpN(pInd,:)); cla;
J = CL; J = J(2:end,:);
hold on;
plot3(X0DC(1,:),X0DC(2,:),X0DC(3,:),'clipping',0,'linewidth',lw,'color',J(1,:));
plot3(X1DC(1,:),X1DC(2,:),X1DC(3,:),'clipping',0,'linewidth',lw,'color',J(2,:));
plot3(X2DC(1,:),X2DC(2,:),X2DC(3,:),'clipping',0,'linewidth',lw,'color',J(3,:));
plot3(X3DC(1,:),X3DC(2,:),X3DC(3,:),'clipping',0,'linewidth',lw,'color',J(4,:));
plot3(-X0DC(1,:),-X0DC(2,:),X0DC(3,:),'clipping',0,'linewidth',lw,'color',[J(1,:),.01]);
plot3(-X1DC(1,:),-X1DC(2,:),X1DC(3,:),'clipping',0,'linewidth',lw,'color',[J(2,:),.01]);
plot3(-X2DC(1,:),-X2DC(2,:),X2DC(3,:),'clipping',0,'linewidth',lw,'color',[J(3,:),.01]);
plot3(-X3DC(1,:),-X3DC(2,:),X3DC(3,:),'clipping',0,'linewidth',lw,'color',[J(4,:),.01]);
for i = 1:4
    plot3(XSS1{i}(1,:,1)*Sc+Sh(1),XSS1{i}(2,:,1)*Sc+Sh(2),XSS1{i}(3,:,1)*Sc+Sh(3),'clipping',0,'linewidth',lw,'color',J(i,:));
end
plot3([XSS1{1}(1,1,1) XSS1{1}(1,1,1)*Sc+Sh(1)],...
      [XSS1{1}(2,1,1) XSS1{1}(2,1,1)*Sc+Sh(2)],...
      [XSS1{1}(3,1,1) XSS1{1}(3,1,1)*Sc+Sh(3)],'k--','clipping',0);
plot3([XSS1{1}(1,end,1) XSS1{1}(1,end,1)*Sc+Sh(1)],...
      [XSS1{1}(2,end,1) XSS1{1}(2,end,1)*Sc+Sh(2)],...
      [XSS1{1}(3,end,1) XSS1{1}(3,end,1)*Sc+Sh(3)],'k--','clipping',0);
axSh = 45; axL = 20;
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
hold off;
set(gca,'visible',0);

% Text and Annotations
text(labX,subp(pInd,4)+.2,'\textbf{e}~~~~training on 1-cycle',NVTitle{:});
text(.07,-.02,'$x_1$',NVTextR{:});
text(.10,.07,'$x_2$',NVTextR{:});
text(-.05,.18,'$x_3$',NVTextR{:});
view(vw);
axis([-45 45 -40 80 10 140]);
% Labels
delete(findall(gcf,'type','annotation'))
annotation('line',[0 .03]+.13, [1 1]*.125,'linewidth',.7,'color',J(1,:));
annotation('line',[0 .03]+.13, [1 1]*.09,'linewidth',.7,'color',J(2,:));
annotation('line',[0 .03]+.13, [1 1]*.055,'linewidth',.7,'color',J(3,:));
annotation('line',[0 .03]+.13, [1 1]*.02,'linewidth',.7,'color',J(4,:));
text(.74,.32,'$\rho$',NVTextH{:});
text(.74,.21,'$100.3$',NVTextH{:});
text(.74,.13,'$100.2$',NVTextH{:});
text(.74,.05,'$100.1$',NVTextH{:});
text(.74,-.03,'$100.0$',NVTextH{:});
drawnow;


pInd = 6;
subplot('position',subpN(pInd,:)); cla;
J = CL; J = J(2:end,:);
hold on;
plot3(X4DC(1,:),X4DC(2,:),X4DC(3,:),'clipping',0,'linewidth',lw,'color',J(5,:));
plot3(XSS2{1}(1,:,1)*Sc+Sh(1),XSS2{1}(2,:,1)*Sc+Sh(2),XSS2{1}(3,:,1)*Sc+Sh(3),'clipping',0,'linewidth',lw,'color',J(5,:));
plot3(XSS2{2}(1,:,1)*Sc+Sh(1),XSS2{2}(2,:,1)*Sc+Sh(2),XSS2{2}(3,:,1)*Sc+Sh(3),'clipping',0,'linewidth',lw,'color',J(5,:));
plot3([XSS2{1}(1,1,1) XSS2{1}(1,1,1)*Sc+Sh(1)],...
      [XSS2{1}(2,1,1) XSS2{1}(2,1,1)*Sc+Sh(2)],...
      [XSS2{1}(3,1,1) XSS2{1}(3,1,1)*Sc+Sh(3)],'k--','clipping',0);
plot3([XSS2{1}(1,end,1) XSS2{1}(1,end,1)*Sc+Sh(1)],...
      [XSS2{1}(2,end,1) XSS2{1}(2,end,1)*Sc+Sh(2)],...
      [XSS2{1}(3,end,1) XSS2{1}(3,end,1)*Sc+Sh(3)],'k--','clipping',0);
plot3([0 axL]-axSh, [0 0]-axSh, [0 0]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 axL]-axSh, [0 0]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
plot3([0 0]-axSh, [0 0]-axSh, [0 axL]-axSh+30, 'k-','linewidth', 1, 'clipping', 0);
hold off;
set(gca,'visible',0);
view(vw);
axis([-45 45 -40 80 10 140]);

% Text and Annotations
text(labX,subp(pInd,4)+.2,'\textbf{f}~~2-cycle (not trained)',NVTitle{:});
% Labels
annotation('line',[0 .03]+.385, [1 1]*.125,'linewidth',.7,'color',J(5,:));
text(.74,.32,'$\rho$',NVTextH{:});
text(.74,.21,'$99.8$',NVTextH{:});
drawnow;


%% Load period doubling data and solve periodicity
load('Lorenz_sweep_data.mat');
load('XPC_XCT.mat');
rhoL = rhoM;

rth = 99.984;
XPO = XPO(rL<rth);
XPO2 = XPO2(rhoL<rth);
rL = rL(rL<rth);
rhoL = rhoL(rhoL<rth);

% Sort
[rL,rLI] = sort(rL,'descend');
[rhoL,rhoLI] = sort(rhoL,'descend');
XPO = XPO(rLI);
XPO2 = XPO2(rhoLI);

PO = zeros(length(rL),1);
PO2 = zeros(length(rhoL),1);
ep = 1e-4;
for i = 1:length(rL)
    xp = sqrt(((XPO{i}(3,1:end-1) - XPO{i}(3,end)).^2));
    a = length(xp)-find(xp<ep,1,'last')+1;
    if(~isempty(a))
        PO(i) = a;
    else
        PO(i) = length(xp);
    end
end
for i = 1:length(rhoL)
    xp = sqrt(((XPO2{i}(3,1:end-1) - XPO2{i}(3,end)).^2));
    a = length(xp)-find(xp<ep,1,'last')+1;
    if(~isempty(a))
        PO2(i) = a;
    else
        PO2(i) = length(xp);
    end
end

PDI = find(diff(log2(PO))==1);
rPDI = mean(rL([PDI PDI+1]),2);


%% g: Plot Full
pInd = 7;
subplot('position',subpN(pInd,:)); cla;
set(gca,'XDir','reverse','visible',0);

pSk1 = round(PO.^1-1);
pSk2 = round(PO2.^1-1);
CP = winter(1000);
rR = [linspace(100,99.5,size(CP,1))];

cnt = 1;
hold on;
ax = [99.42 100.5 94.8 96.65];
xtick = 100.4; ytick = ax(3) + diff(ax(3:4))*.16; 
% Doubling Lines
line([1;1].*rPDI(1:2)',[ytick; ax(4)].*ones(1,2),'color',[1 1 1]*.7,'linewidth',.5,'linestyle',':');

for i = 1:length(rL)
    if(cnt < pSk1(i))
        cnt = cnt + 1;
    else
        plot(rL(i),XPO{i}(3,end-PO(i):end),'k.','markersize',15);
        cnt = 1;
    end
end
axis(ax);
drawnow;

% Box
ax2 = [99.52 99.6 95.5 95.83];
xtick2 = 99.582; ytick2 = ax2(3) + diff(ax2(3:4))*.16; 
ax2b = [ax2(1) xtick2 ytick2 ax2(4)];
line(ax2b([1 2 2 1 1 2]), ax2b([3 3 4 4 3 3]),'color','k','linewidth',.5);
line([0 -.5] + ax2b(2), [0 .8] + ax2b(4),'color','k','linewidth',.5','linestyle','--','clipping',0);
line([0 -.5] + ax2b(2), [0 -.45] + ax2b(3),'color','k','linewidth',.5','linestyle','--','clipping',0);

cnt = 1;
for i = 1:length(rhoL)
    if(cnt < pSk1(i))
        cnt = cnt + 1;
    else
        [~,a] = min(abs(rhoL(i)-rR));
        plot(rhoL(i),XPO2{i}(3,end-PO2(i):end),'r.','markersize',7.5,'color',CP(a,:));
        cnt = 1;
    end
end

for i = 1:length(rhoV)-1
    plot(rhoV(i),XPE{i}(3,end),'.','markersize',20,'color',CL(i,:));
end

% Axis
xtickL = [100.3 100 99.7 99.5];
ytickL = [95.5 96.5];
tickw = diff(ax(1:2))*.02;
tickh = diff(ax(3:4))*.02;
line([ax(1) xtick], [1 1]*ytick, 'color','k','linewidth',.5);
line([1 1]*xtick, [ytick ax(4)], 'color','k','linewidth',.5);
line([1;1].*xtickL, [ytick;ytick+tickh].*ones(1,length(xtickL)),'color','k','linewidth',.5);
line([xtick-tickw;xtick].*ones(1,length(ytickL)), [1;1].*ytickL,'color','k','linewidth',.5);

% Text
for i = 1:length(xtickL)
    text(xtickL(i),ytick-diff(ax(3:4))*.06,num2str(xtickL(i)),NVTextHD{:});
    text(xtickL(i),ytick-diff(ax(3:4))*.15,num2str((100.3-xtickL(i))/.1),NVTextHD{:});
end
text(xtick+.12,ytick-diff(ax(3:4))*.06,'$\rho:$',NVTextRD{:});
text(xtick+.12,ytick-diff(ax(3:4))*.15,'$c:$',NVTextRD{:});
for i = 1:length(ytickL)
    text(xtick+.17,ytickL(i),num2str(ytickL(i)),NVTextRD{:});
end
text(xtick+.12,mean(ytickL),'$z$',NVTextRD{:});
text(100.33,96.1,'$2^0$-cycle',NVTextRD{:},'color',[1 1 1]*.7);
text(99.84,96.1,'$2^1$',NVTextRD{:},'color',[1 1 1]*.7);
text(99.62,96.1,'$2^2$',NVTextRD{:},'color',[1 1 1]*.7);

% Legend
plot(100.18,95.5,'k.','markersize',20);
plot(100.25,95.2,'k.','markersize',7.5,'color',CP(1,:));
plot(100.18,95.2,'k.','markersize',7.5,'color',CP(round(size(CP,1)/2),:));
plot(100.11,95.2,'k.','markersize',7.5,'color',CP(end,:));
text(100.18,95.62,'true',NVTextHD{:});
text(100.18,95.3,'predicted',NVTextHD{:});

drawnow;
hold off;

% Text and Annotations
text(labX,subp(pInd,4)+.2,'\textbf{g}~~~~true and predicted period doubling bifurcation diagram',NVTitle{:});


%% g: Plot Section
rTh = 99.58;
pSk1 = round(PO.^1-1);
rThI = find(rL<rTh);
rhoThI = find(rhoL<rTh);

pInd = 8;
subplot('position',subpN(pInd,:)); cla;
set(gca,'XDir','reverse','visible',0);


hold on;
% Doubling Lines
line([1;1].*rPDI(3:4)',[ytick2; ax2(4)].*ones(1,2),'color',[1 1 1]*.7,'linewidth',.5,'linestyle',':');
cnt = 1;
for i = 1:length(rThI)
    if(cnt < pSk1(i))
        cnt = cnt + 1;
    else
        XPOP = XPO{rThI(i)}(3,end-PO(rThI(i)):end);
        XPOP = XPOP(XPOP <96 & XPOP > 95.5);
        plot(rL(rThI(i)),XPOP,'k.','markersize',20);
        cnt = 1;
    end
end
axis(ax2);
drawnow;

cnt = 1;
for i = 1:length(rhoThI)
    if(cnt < pSk1(i))
        cnt = cnt + 1;
    else
        [~,a] = min(abs(rhoL(rhoThI(i))-rR));
        XPOP = XPO2{rhoThI(i)}(3,end-PO2(rhoThI(i)):end);
        XPOP = XPOP(XPOP <96 & XPOP > 95.5);
        plot(rhoL(rhoThI(i)),XPOP,'r.','markersize',7.5,'color',CP(a,:));
    end
end

% Axis
xtickL = [99.53 99.57];
ytickL = [95.6 95.8];
tickw = diff(ax2(1:2))*.02;
tickh = diff(ax2(3:4))*.02;
line([ax2(1) xtick2], [1 1]*ytick2, 'color','k','linewidth',.5);
line([1 1]*xtick2, [ytick2 ax2(4)], 'color','k','linewidth',.5);
line([1;1].*xtickL, [ytick2;ytick2+tickh].*ones(1,length(xtickL)),'color','k','linewidth',.5);
line([xtick2-tickw;xtick2].*ones(1,length(ytickL)), [1;1].*ytickL,'color','k','linewidth',.5);

text(99.565,95.68,'$2^2$',NVTextRD{:},'color',[1 1 1]*.7);
text(99.54,95.7,'$2^3$',NVTextRD{:},'color',[1 1 1]*.7);
text(99.529,95.71,'$2^4$',NVTextRD{:},'color',[1 1 1]*.7);
hold off;

% Text
for i = 1:length(xtickL)
    text(xtickL(i),ytick2-diff(ax2(3:4))*.06,num2str(xtickL(i)),NVTextHD{:});
    text(xtickL(i),ytick2-diff(ax2(3:4))*.15,num2str((100.3-xtickL(i))/.1),NVTextHD{:});
end
text(xtick+.12,ytick-diff(ax(3:4))*.06,'$\rho:$',NVTextRD{:});
text(xtick+.12,ytick-diff(ax(3:4))*.15,'$c:$',NVTextRD{:});
for i = 1:length(ytickL)
    text(xtick+.17,ytickL(i),num2str(ytickL(i)),NVTextRD{:});
end

drawnow;


%% Save
fName = 'fig_bifurcate_period_double';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');