%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(5); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 9.75];                      % Figure Size in cm  [w,h]
fMarg = [.4 .4 .2 .2];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 5.00 4.75  4.75];...     % Subplot position in cm [x,y,w,h]
        [ 4.75 5.00 4.75  4.75];...
        [ 9.50 5.00 4.75  4.75];...
        [14.25 5.00 4.75  4.75];...
        [ 0.00 0.00 4.75  4.75];...
        [ 4.75 0.00 4.75  4.75];...
        [ 9.50 0.00 9.50  4.75]];

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
NVTextLD = {'Units','Data','fontsize',FS,'HorizontalAlignment','right'};

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


%% Saddle-Node normal form time series
disp('Simulating Attractor');

% Saddle Node
nPLSN = linspace(-.01,-.001,4);
x0aSN = -10; x0bSN = .03;
HSN = saddlenode(x0aSN, delT, [-1 -1]);
X0aSN = zeros(1,n,4,length(nPLSN));
X0bSN = zeros(1,n,4,length(nPLSN));
for i = 1:length(nPLSN)
    HSN.parms = [nPLSN(i) -1]; HSN.x = x0aSN;
    X0aSN(:,:,:,i) = HSN.propagate(n);
    HSN.parms = [nPLSN(i) -1]; HSN.x = x0bSN;
    X0bSN(:,:,:,i) = HSN.propagate(n);
end

% Supercritical Pitchfork
nPLSP = linspace(-.1,-.01,4);
x0SP = -10;
HSP = pitchfork(x0SP, delT, [-1 -1]);
X0aSP = zeros(1,n,4,length(nPLSP));
X0bSP = zeros(1,n,4,length(nPLSP));
for i = 1:length(nPLSP)
    HSP.parms = [nPLSP(i) -1]; HSP.x = x0SP;
    X0aSP(:,:,:,i) = HSP.propagate(n);
    HSP.parms = [nPLSP(i) -1]; HSP.x = -x0SP;
    X0bSP(:,:,:,i) = HSP.propagate(n);
end

sig = 0.5;
c = 0.001;
gam = 10;


%% Initialize reservoir constant parameters
N = 50;                                         % Number of reservoir states
p = 0.1;                                        % Reservoir initial densitys
% Equilibrium point
x0 = zeros(size(X0aSN,1),1);
c0 = zeros(length(c),1);


%% Initial reservoir random parameters
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);       % Stabilize base matrix
% Input matrices
B = 2*(rand(N,size(X0aSN,1))-.5)*sig;
C = 2*(rand(N,length(c))-.5)*c;
% Fixed point
r0 = (rand(N,1)*.2+.8).* sign(rand(N,1)-0.5);   % Distribution of offset


%% Create reservoir object
R2 = ReservoirTanh(A,B,C,r0,x0,c0,delT,gam);


%% Train reservoir
Cin = ones(1,n,4);

% Drive reservoir
disp('Simulating Reservoir');
RTaSN = zeros(N,n,length(nPLSN));
RTbSN = zeros(N,n,length(nPLSN));
RTaSP = zeros(N,n,length(nPLSP));
RTbSP = zeros(N,n,length(nPLSP));
for i = 1:length(nPLSN)
    RTaSN(:,:,i) = R2.train(X0aSN(:,:,:,i),(i-1)*Cin);
    RTbSN(:,:,i) = R2.train(X0bSN(:,:,:,i),(i-1)*Cin);
    RTaSP(:,:,i) = R2.train(X0aSP(:,:,:,i),(i-1)*Cin);
    RTbSP(:,:,i) = R2.train(X0bSP(:,:,:,i),(i-1)*Cin);
end

% Reshape input and reservoir data for training
RTTaSN = reshape(RTaSN(:,ind_t,:),[N,length(nPLSN)*n_t]);
XTTaSN = reshape(X0aSN(:,ind_t,1,:),[size(X0aSN,1),length(nPLSN)*n_t]);
RTTbSN = reshape(RTbSN(:,ind_t,:),[N,length(nPLSN)*n_t]);
XTTbSN = reshape(X0bSN(:,ind_t,1,:),[size(X0aSN,1),length(nPLSN)*n_t]);
RTTaSP = reshape(RTaSP(:,ind_t,:),[N,length(nPLSP)*n_t]);
XTTaSP = reshape(X0aSP(:,ind_t,1,:),[size(X0aSP,1),length(nPLSP)*n_t]);
RTTbSP = reshape(RTbSP(:,ind_t,:),[N,length(nPLSP)*n_t]);
XTTbSP = reshape(X0bSP(:,ind_t,1,:),[size(X0aSP,1),length(nPLSP)*n_t]);

disp('Training W');
WSN = lsqminnorm([RTTaSN RTTbSN]',[XTTaSN XTTbSN]')';
nErrSN = norm(WSN*[RTTaSN RTTbSN] - [XTTaSN XTTbSN]);
disp(['Training error: ' num2str(nErrSN)]);
WSP = lsqminnorm([RTTaSP RTTbSP]',[XTTaSP XTTbSP]')';
nErrSP = norm(WSP*[RTTaSP RTTbSP] - [XTTaSP XTTbSP]);
disp(['Training error: ' num2str(nErrSP)]);

% clear RTTa RTTb XTTa XTTb
clear RTTaSN XTTaSN RTTbSN XTTbSN
clear RTTaSP XTTaSP RTTbSP XTTbSP


%% Evaluate Fixed Points
% Functions
MSN = A + B*WSN;
MSP = A + B*WSP;
d = R2.d; I = eye(N);
% Jacobian of fixed point function
JSN = @(rv,cv) -I + (1-tanh(MSN*rv+C*cv+d).^2).*MSN;
JSP = @(rv,cv) -I + (1-tanh(MSP*rv+C*cv+d).^2).*MSP;
% Quadratic cost of fixed point evaluation
ESN = @(rv,cv) (-rv + tanh(MSN*rv + C*cv + d))'*(-rv + tanh(MSN*rv + C*cv + d));
ESP = @(rv,cv) (-rv + tanh(MSP*rv + C*cv + d))'*(-rv + tanh(MSP*rv + C*cv + d));
% Reservoir equation
rdotSN = @(rv,cv) -rv + tanh(MSN*rv + C*cv + d);
rdotSP = @(rv,cv) -rv + tanh(MSP*rv + C*cv + d);

nrT = .01;
crT = (nrT - nPLSP(1)) / mean(diff(nPLSP));

% Define fixed points
xs1SN = -sqrt(-nPLSN(1));   XS1SN = repmat(xs1SN,[1,n,4]);
xs2SN =  sqrt(-nPLSN(1));   XS2SN = repmat(xs2SN,[1,n,4]);
xs1SP = 0;                  XS1SP = repmat(xs1SP,[1,n,4]);
xs2SP = -sqrt(nrT);         XS2SP = repmat(xs2SP,[1,n,4]);
xs3SP =  sqrt(nrT);         XS3SP = repmat(xs3SP,[1,n,4]);
xs4SP = 0;                  XS4SP = repmat(xs4SP,[1,n,4]);

% FP1: c = 0
disp('Identifying guesses for unseen reservoir fixed points');
R2.train(XS1SN,0*Cin);      rs1SN = R2.r;
R2.train(XS2SN,0*Cin);      rs2SN = R2.r;
R2.train(XS1SP,0*Cin);      rs1SP = R2.r;
R2.train(XS2SP,crT*Cin);    rs2SP = R2.r;
R2.train(XS3SP,crT*Cin);    rs3SP = R2.r;
R2.train(XS4SP,crT*Cin);    rs4SP = R2.r;


%% Parameter ranges for fp evaluation
nrV = 1000;
% Parameter values to sweep
rVrSN = [linspace(nPLSN(1),-.1,nrV) linspace(-.1,-.00001,nrV)];
rVr1SP = [linspace(nPLSP(1),-1,nrV) linspace(-1,-.001,nrV)];
rVr2SP = [linspace(nrT,0.005,nrV) linspace(0.005,1,nrV)];
% Map parameter values to c values for controlling reservoir
cVrSN = (rVrSN - nPLSN(1)) / mean(diff(nPLSN));
cVr1SP = (rVr1SP - nPLSP(1)) / mean(diff(nPLSP));
cVr2SP = (rVr2SP - nPLSP(1)) / mean(diff(nPLSP));
% Save reservoir fixed points
rVM1SN = zeros(N,length(rVrSN)); rVM1SN(:,1) = rs1SN;
rVM2SN = zeros(N,length(rVrSN)); rVM2SN(:,1) = rs2SN;
rVM1SP = zeros(N,length(rVr1SP)); rVM1SP(:,1) = rs1SP;
rVM2SP = zeros(N,length(rVr2SP)); rVM2SP(:,1) = rs2SP;
rVM3SP = zeros(N,length(rVr2SP)); rVM3SP(:,1) = rs3SP;
rVM4SP = zeros(N,length(rVr2SP)); rVM4SP(:,1) = rs4SP;
% Evaluate quadratic cost
EV1SN = zeros(length(rVrSN),1);
EV2SN = zeros(length(rVrSN),1);
EV1SP = zeros(length(rVr1SP),1);
EV2SP = zeros(length(rVr1SP),1);
EV3SP = zeros(length(rVr1SP),1);
EV4SP = zeros(length(rVr1SP),1);
% Store real component of largest real eigenvalue
eV1SN = zeros(length(rVrSN),1);
eV2SN = zeros(length(rVrSN),1);
eV1SP = zeros(length(rVr1SP),1);
eV2SP = zeros(length(rVr1SP),1);
eV3SP = zeros(length(rVr1SP),1);
eV4SP = zeros(length(rVr1SP),1);

disp('Tracking feedback reservoir fixed points');
for i = 2:length(rVrSN)
    r0v1SN = rVM1SN(:,i-1);
    r0v2SN = rVM2SN(:,i-1);
    r0v1SP = rVM1SP(:,i-1);
    r0v2SP = rVM2SP(:,i-1);
    r0v3SP = rVM3SP(:,i-1);
    r0v4SP = rVM4SP(:,i-1);
    for j = 1:10
        r0v1pSN = -JSN(r0v1SN,cVrSN(i))\rdotSN(r0v1SN,cVrSN(i));
        r0v2pSN = -JSN(r0v2SN,cVrSN(i))\rdotSN(r0v2SN,cVrSN(i));
        r0v1pSP = -JSP(r0v1SP,cVr1SP(i))\rdotSP(r0v1SP,cVr1SP(i));
        r0v2pSP = -JSP(r0v2SP,cVr2SP(i))\rdotSP(r0v2SP,cVr2SP(i));
        r0v3pSP = -JSP(r0v3SP,cVr2SP(i))\rdotSP(r0v3SP,cVr2SP(i));
        r0v4pSP = -JSP(r0v4SP,cVr2SP(i))\rdotSP(r0v4SP,cVr2SP(i));
        r0v1SN = r0v1pSN + r0v1SN;
        r0v2SN = r0v2pSN + r0v2SN;
        r0v1SP = r0v1pSP + r0v1SP;
        r0v2SP = r0v2pSP + r0v2SP;
        r0v3SP = r0v3pSP + r0v3SP;
        r0v4SP = r0v4pSP + r0v4SP;
    end
    rVM1SN(:,i) = r0v1SN;
    rVM2SN(:,i) = r0v2SN;
    rVM1SP(:,i) = r0v1SP;
    rVM2SP(:,i) = r0v2SP;
    rVM3SP(:,i) = r0v3SP;
    rVM4SP(:,i) = r0v4SP;
    EV1SN(i) = ESN(r0v1SN,cVrSN(i));
    EV2SN(i) = ESN(r0v2SN,cVrSN(i));
    EV1SP(i) = ESP(r0v1SP,cVr1SP(i));
    EV2SP(i) = ESP(r0v2SP,cVr2SP(i));
    EV3SP(i) = ESP(r0v3SP,cVr2SP(i));
    EV4SP(i) = ESP(r0v4SP,cVr2SP(i));
    eV1SN(i) = max(real(eig(JSN(r0v1SN,cVrSN(i)))));
    eV2SN(i) = max(real(eig(JSN(r0v2SN,cVrSN(i)))));
    eV1SP(i) = max(real(eig(JSP(r0v1SP,cVr1SP(i)))));
    eV2SP(i) = max(real(eig(JSP(r0v2SP,cVr2SP(i)))));
    eV3SP(i) = max(real(eig(JSP(r0v3SP,cVr2SP(i)))));
    eV4SP(i) = max(real(eig(JSP(r0v4SP,cVr2SP(i)))));
end
% Forget initial fixed points
rVM1SN = rVM1SN(:,nrV+1:end);
rVM2SN = rVM2SN(:,nrV+1:end);
rVM1SP = rVM1SP(:,nrV+1:end);
rVM2SP = rVM2SP(:,nrV+1:end);
rVM3SP = rVM3SP(:,nrV+1:end);
rVM4SP = rVM4SP(:,nrV+1:end);
% Forget initial parameter values that were swept
rVrSN = rVrSN(nrV+1:end);
rVr1SP = rVr1SP(nrV+1:end);
rVr2SP = rVr2SP(nrV+1:end);
% Forget initial eigenvalues
eV1SN = eV1SN(nrV+1:end);
eV2SN = eV2SN(nrV+1:end);
eV1SP = eV1SP(nrV+1:end);
eV2SP = eV2SP(nrV+1:end);
eV3SP = eV3SP(nrV+1:end);
eV4SP = eV4SP(nrV+1:end);


%% Plot
pInd = 1;
subplot('position',subpN(pInd,:)); cla;
axL = [n_t*[-.01 -.01 .98]; .4 -.36 -.36];

hold on;
line([axL(1,1) n_t],[0 0],'linewidth',.3,'linestyle','--','color',[1 1 1]*.8);
for i = 1:4
    plot(1:n_t,X0aSN(:,ind_t,1,i),'linewidth',.7,'color',CL(i+1,:));
    plot(1:n_t,X0bSN(:,ind_t,1,i),'linewidth',.7,'color',CL(i+1,:));
    line(n_t*[.5 .65], [1 1]*.27-.06*(i-1),'linewidth',1,'color',CL(i+1,:));
    text(n_t*.7,.27-.06*(i-1),num2str(nPLSN(i)),NVTextRD{:});
end
line(axL(1,:),axL(2,:),'linewidth',.7,'color','k');
line(axL(1,1)+[0 .03]*(axL(1,3)-axL(1,2)),-.3*[1 1],'linewidth',.7,'color','k');
line(axL(1,1)+[0 .03]*(axL(1,3)-axL(1,2)),.3*[1 1],'linewidth',.7,'color','k');
plot_spline([n_t*[-.01 1];[1 1]*axL(2,end)],'head',1,'headpos',1,'linewidth',.01);
hold off;
% Text
text(labX,subp(pInd,4),'\textbf{a}~~~~~~training time series',NVTitle{:});
text(-.1*n_t,0,'$x$',NVTextLD{:});
text(-.018*n_t,.3,'$.3$',NVTextLD{:});
text(-.018*n_t,-.3,'$-.3$',NVTextLD{:});
text(mean(axL(1,2:3)),-.44,'$t$',NVTextLD{:});
text(n_t*.8,.35,'$a$',NVTextRD{:});
% Plot parameters
set(gca,'visible',0);
axis([-.05*n_t n_t -.43 .43]);


pInd = 2;
subplot('position',subpN(pInd,:)); cla;
colormap bone;
axL2 = [-0.11 -0.11 0.01; axL(2,:)];
hold on;
scatter(rVrSN,WSN*rVM1SN,20,eV1SN,'filled');
scatter(rVrSN,WSN*rVM2SN,20,eV2SN,'filled');
plot(rVrSN,sqrt(-rVrSN),'k--','linewidth',1);
plot(rVrSN,-sqrt(-rVrSN),'k-','linewidth',1);
line(axL2(1,:),axL2(2,:),'linewidth',.7,'color','k');
line(axL2(1,1)+[0 .03]*(axL2(1,3)-axL2(1,2)),-.3*[1 1],'linewidth',.7,'color','k');
line(axL2(1,1)+[0 .03]*(axL2(1,3)-axL2(1,2)),.3*[1 1],'linewidth',.7,'color','k');
line([1 1]*-.1,axL2(2,2)+[0 .03]*(axL2(2,1)-axL2(2,2)),'linewidth',.7,'color','k');
line([1 1]*0,axL2(2,2)+[0 .03]*(axL2(2,1)-axL2(2,2)),'linewidth',.7,'color','k');
line([-.105 -.085],[1 1]*.08,'color','k','linestyle','-','linewidth',1);
line([-.105 -.085],[1 1]*.00,'color','k','linestyle','--','linewidth',1);
scatter(linspace(-.103,-.088,3),[1 1 1]*-.08,20,[min(eV1SN);0;max(eV2SN)],'filled');
scatter(ones(1,100)*0.015,linspace(-.2, .2,100),60,linspace(-.1, .1,100)','filled','marker','s');
for i = 1:length(nPLSN)
    plot_spline([[1 1]*nPLSN(i); -sqrt(-nPLSN(i))+[-.12 -.05]],...
        'head',1,'headpos',1,'headwidth',2,'linewidth',.3,'color',CL(i+1,:));
end
% Text
text(labX,subp(pInd,4),'\textbf{b}~~~~~~bifurcation diagram',NVTitle{:});
text(-.12,0,'$x$',NVTextLD{:});
text(-.111,.3,'$.3$',NVTextLD{:});
text(-.111,-.3,'$-.3$',NVTextLD{:});
text(-.1,-.41,'$-.1$',NVTextHD{:});
text(0,-.41,'$0$',NVTextHD{:});
text(-.08,.08,'real stable',NVTextRD{:});
text(-.08,.0,'real unstable',NVTextRD{:});
text(-.08,-.08,'predicted',NVTextRD{:});
text(-.043,-.44,'$a$',NVTextLD{:});
text(.027,0,'reservoir eigenvalue',NVTextHD{:},'rotation',90);
text(0.02,-.27,'$-.1$',NVTextLD{:});
text(0.02,.27,'$.1$',NVTextLD{:});
text(-.025,-.27,'training',NVTextHD{:},'color',CL(3,:).^.3);
hold off;
% Plot parameters
set(gca,'visible',0);
axis([-0.11 0.03 -.43 .43]);
caxis([-0.15,0.15]);


pInd = 3;
subplot('position',subpN(pInd,:)); cla;
axL = [n_t*[-.01 -.01 .98]; 1.2 -1.18 -1.18];
hold on;
line([axL(1,1) n_t],[0 0],'linewidth',.3,'linestyle','--','color',[1 1 1]*.8);
for i = 1:4
    plot(1:n_t,X0aSP(:,ind_t,1,i),'linewidth',.7,'color',CL(i+1,:));
    plot(1:n_t,X0bSP(:,ind_t,1,i),'linewidth',.7,'color',CL(i+1,:));
    line(n_t*[.5 .65], [1 1]*.85-.2*(i-1),'linewidth',1,'color',CL(i+1,:));
    text(n_t*.7,.85-.2*(i-1),num2str(nPLSP(i)),NVTextRD{:});
end
line(axL(1,:),axL(2,:),'linewidth',.7,'color','k');
line(axL(1,1)+[0 .03]*(axL(1,3)-axL(1,2)),-1*[1 1],'linewidth',.7,'color','k');
line(axL(1,1)+[0 .03]*(axL(1,3)-axL(1,2)),1*[1 1],'linewidth',.7,'color','k');
plot_spline([n_t*[-.01 1];[1 1]*axL(2,end)],'head',1,'headpos',1,'linewidth',.01);
hold off;
% Text
text(labX,subp(pInd,4),'\textbf{c}~~~~~~training time series',NVTitle{:});
text(-.1*n_t,0,'$x$',NVTextLD{:});
text(-.018*n_t,1,'$1$',NVTextLD{:});
text(-.018*n_t,-1,'$-1$',NVTextLD{:});
text(mean(axL(1,2:3)),-1.43,'$t$',NVTextLD{:});
text(n_t*.8,1.1,'$a$',NVTextRD{:});
% Plot parameters
set(gca,'visible',0);
axis([-.05*n_t n_t -1.4 1.4]);


pInd = 4;
subplot('position',subpN(pInd,:)); cla;
colormap bone;
axL2 = [-1.1 -1.1 1.1; axL(2,:)];
hold on;
scatter(rVr1SP,WSP*rVM1SP,20,eV1SP,'filled');
scatter(rVr2SP,WSP*rVM2SP,20,eV2SP,'filled');
scatter(rVr2SP,WSP*rVM3SP,20,eV3SP,'filled');
scatter(rVr2SP,WSP*rVM4SP,20,eV4SP,'filled');
plot(rVr1SP,zeros(length(rVr1SP),1),'k-','linewidth',1);
plot(rVr2SP,sqrt(rVr2SP),'k-','linewidth',1);
plot(rVr2SP,-sqrt(rVr2SP),'k-','linewidth',1);
plot(rVr2SP,zeros(length(rVr1SP),1),'k--','linewidth',1);
line(axL2(1,:),axL2(2,:),'linewidth',.7,'color','k');
line(axL2(1,1)+[0 .03]*(axL2(1,3)-axL2(1,2)),-1*[1 1],'linewidth',.7,'color','k');
line(axL2(1,1)+[0 .03]*(axL2(1,3)-axL2(1,2)),1*[1 1],'linewidth',.7,'color','k');
line([1 1]*-1,axL2(2,2)+[0 .03]*(axL2(2,1)-axL2(2,2)),'linewidth',.7,'color','k');
line([1 1]*1,axL2(2,2)+[0 .03]*(axL2(2,1)-axL2(2,2)),'linewidth',.7,'color','k');
for i = 1:length(nPLSN)
    plot_spline([[1 1]*nPLSP(i); [-.6 -.25]],...
        'head',1,'headpos',1,'headwidth',2,'linewidth',.3,'color',CL(i+1,:));
end
% Text
text(labX,subp(pInd,4),'\textbf{d}~~~~~~bifurcation diagram',NVTitle{:});
text(-1.3,0,'$x$',NVTextLD{:});
text(-1.12,1,'$1$',NVTextLD{:});
text(-1.12,-1,'$-1$',NVTextLD{:});
text(-1,-1.35,'$-1$',NVTextHD{:});
text(1,-1.35,'$1$',NVTextHD{:});
text(.1,-1.43,'$a$',NVTextLD{:});
text(-.1,-.75,'training',NVTextHD{:},'color',CL(3,:).^.3);
hold off;
% Plot parameters
set(gca,'visible',0);
axis([-1.2 1.2 -1.4 1.4]);
caxis([-0.15,0.15]);
drawnow;


%% Jansen linkage
% Create Linkage
Am = 38;    Bm = 40.5;  Cm = 40.3;  Dm = 35.1;  Em = 65;
Fm = 46;    Gm = 37;    Hm = 65.7;  Im = 50;    Jm = 50;
Km = 56;    Lm = 5;     Mm = 14;

% Functions for computing node positions
x = @(r1,r2,d) (d^2-r2^2+r1^2)/(2*d);
y = @(r1,r2,d) sqrt((-d+r2-r1)*(-d-r2+r1)*(-d+r2+r1)*(d+r2+r1))/d/2;
R = @(th) [cosd(th) -sind(th);...
           sind(th)  cosd(th)];
CP = @(r1,r2,C,f) R(atan2d(C(2),C(1)))*...
                  [x(r1,r2,sqrt(sum(C.^2)));...
                   f*y(r1,r2,sqrt(sum(C.^2)))];

% Node positions and edge connectivity for 4 training examples
Bm = 40.5;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
Xs0 = [BA,LM,MJ,JB,ED,FG,CK,HI];

Bm = 40.55;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
Xs1 = [BA,LM,MJ,JB,ED,FG,CK,HI];

Bm = 40.6;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
Xs2 = [BA,LM,MJ,JB,ED,FG,CK,HI];

Bm = 40.65;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
Xs3 = [BA,LM,MJ,JB,ED,FG,CK,HI];

% Node positions for the test examples
Bm = 39.25;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
XsTa = [BA,LM,MJ,JB,ED,FG,CK,HI];

Bm = 41.75;
BA = [0;0];     AL = [Am;0];     LM = [Am;Lm];     MJ = [Am;Lm+Mm];
JB = CP(Bm,Jm,MJ,1);      ED = CP(Dm,Em,JB,1);      CK = CP(Cm,Km,MJ,-1);
FG = CP(Fm,Gm,CK-ED,-1)+ED;       HI = CP(Hm,Im,CK-FG,-1)+FG;
XsTb = [BA,LM,MJ,JB,ED,FG,CK,HI];

% Linkage connectivity
conn = [1 2; 2 3; 3 4; 4 1; 4 5; 5 1; 5 6; 6 7; 7 1; 7 3; 6 8; 7 8];

% Simulate network
n = 100000;                 % Number of simulation steps
ds = 0.05;                  % Step size for simulation
disp('Simulating linkage trajectory');
[XC0,~] = sim_motion(Xs0,[],conn,ds,n,-Xs0,0);
[XC1,~] = sim_motion(Xs1,[],conn,ds,n,-Xs1,0);
[XC2,~] = sim_motion(Xs2,[],conn,ds,n,-Xs2,0);
[XC3,~] = sim_motion(Xs3,[],conn,ds,n,-Xs3,0);
[XCTa,~] = sim_motion(XsTa,[],conn,ds,n,-XsTa,0);
[XCTb,~] = sim_motion(XsTb,[],conn,ds,n,-XsTb,0);

XC0 = XC0(:,:,:,1);
XC1 = XC1(:,:,:,1);
XC2 = XC2(:,:,:,1);
XC3 = XC3(:,:,:,1);
XCTa = XCTa(:,:,:,1);
XCTb = XCTb(:,:,:,1);

% Translate and rotate motion to pin nodes in place
nA = 15;
for i = 1:size(XC0,3)
    XC0(:,:,i) = R(nA+atan2d(XC0(2,2,i)-XC0(2,1,i),XC0(1,2,i)-XC0(1,1,i)))'*(XC0(:,:,i)-XC0(:,1,i));
    XC1(:,:,i) = R(nA+atan2d(XC1(2,2,i)-XC1(2,1,i),XC1(1,2,i)-XC1(1,1,i)))'*(XC1(:,:,i)-XC1(:,1,i));
    XC2(:,:,i) = R(nA+atan2d(XC2(2,2,i)-XC2(2,1,i),XC2(1,2,i)-XC2(1,1,i)))'*(XC2(:,:,i)-XC2(:,1,i));
    XC3(:,:,i) = R(nA+atan2d(XC3(2,2,i)-XC3(2,1,i),XC3(1,2,i)-XC3(1,1,i)))'*(XC3(:,:,i)-XC3(:,1,i));
    XCTa(:,:,i) = R(nA+atan2d(XCTa(2,2,i)-XCTa(2,1,i),XCTa(1,2,i)-XCTa(1,1,i)))'*(XCTa(:,:,i)-XCTa(:,1,i));
    XCTb(:,:,i) = R(nA+atan2d(XCTb(2,2,i)-XCTb(2,1,i),XCTb(1,2,i)-XCTb(1,1,i)))'*(XCTb(:,:,i)-XCTb(:,1,i));
end
XC0 = XC0 + [0;80];
XC1 = XC1 + [0;80];
XC2 = XC2 + [0;80];
XC3 = XC3 + [0;80];
XCTa = XCTa + [0;80];
XCTb = XCTb + [0;80];
XCr0 = squeeze(XC0(:,end,:,1));
XCr1 = squeeze(XC1(:,end,:,1));
XCr2 = squeeze(XC2(:,end,:,1));
XCr3 = squeeze(XC3(:,end,:,1));
XCrTa = squeeze(XCTa(:,end,:,1));
XCrTb = squeeze(XCTb(:,end,:,1));


%% Draw mechanism
lw = 0.7;

pInd = 5;
subplot('position',subpN(pInd,:)); cla;
lI = 4;
plI1 = 700; plI2 = 1500;
Lc = zeros(size(conn,1),3); Lc(lI,:) = [1 1 1];
plot(XCr0(1,:),XCr0(2,:),'linewidth',.5,'color',CL(2,:),'clipping',0);
visualize_network(XC0(:,:,plI1,1),[],conn,'lalpha',.2,'lcolor',Lc,...
                  'lwidth',lw,'bwidth',lw,'scolor',[1 1 1]);
visualize_network(XC0(:,1:2,plI2,1),XC0(:,3:end,plI2,1),conn,'scolor',[0 0 0],...
                  'lcolor',Lc,'lwidth',lw,'bwidth',lw,'ucolor',[1 1 1]);
line(XC0(1,conn(lI,:),plI1),XC0(2,conn(lI,:),plI1),'linestyle','--','color',[CL(2,:) .2]);
line(XC0(1,conn(lI,:),plI2),XC0(2,conn(lI,:),plI2),'linestyle','--','color',CL(2,:));
set(gca,'visible',0);
axis([[-1 1]*80-10 -10 150]);
% Text
text(labX,subp(pInd,4)-.2,'\textbf{e}~~~~~~~~~Jansen linkage',NVTitle{:});


% Legend
pInd = 6;
subplot('position',subpN(pInd,:)); cla;
axis([0 1 0 1]);
visualize_network([.0;.8],[],[1 1],'lwidth',lw,'bwidth',lw,'scolor',[1 1 1]);
visualize_network([.0;.7],[],[1 1],'lwidth',lw,'bwidth',lw,'scolor',[0 0 0]);
line([0 .2]-.1, [1 1]*.6,'linewidth',lw,'color','k','clipping',0);
line([0 .2]-.1, [1 1]*.505,'linewidth',lw,'color',CL(2,:),'clipping',0);
line([0 .2]-.1, [1 1]*.485,'linewidth',lw,'color',CL(2,:),'clipping',0);
line([0 .2]-.1, [1 1]*.465,'linewidth',lw,'color',CL(4,:),'clipping',0);
line([0 .2]-.1, [1 1]*.445,'linewidth',lw,'color',CL(5,:),'clipping',0);
line([0 .2]-.1, [1 1]*.35,'linewidth',lw,'color',CL(2,:),'linestyle','--','clipping',0);
line([0 .2]-.1, [1 1]*.25,'linewidth',lw,'color',CL(3,:),'linestyle','--','clipping',0);
line([0 .2]-.1, [1 1]*.15,'linewidth',lw,'color',CL(4,:),'linestyle','--','clipping',0);
line([0 .2]-.1, [1 1]*.05,'linewidth',lw,'color',CL(5,:),'linestyle','--','clipping',0);
% Text
text(0.15,0.8,'free node',NVTextRD{:});
text(0.15,0.7,'pinned node',NVTextRD{:});
text(0.15,0.6,'fixed edge length',NVTextRD{:});
text(0.15,0.475,'training trajectory',NVTextRD{:});
text(0.15,0.35,'edge length = $40.5$',NVTextRD{:});
text(0.15,0.25,'edge length = $40.55$',NVTextRD{:});
text(0.15,0.15,'edge length = $40.6$',NVTextRD{:});
text(0.15,0.05,'edge length = $40.65$',NVTextRD{:});
set(gca,'visible',0);


%% Train reservoir
delT = 0.001;                       % Simulation Time-Step
n = size(XCr0,2);
n_w = 0.5*n;
n_t = n-n_w;
ind_t = (1:n_t) + n_w;              % Index of training samples

N = 900;                                    % Number of reservoir states
sig = 0.7;
c = 0.013;
gam = 1;
p = 0.1;                                    % Reservoir initial density
A = (rand(N) - .5)*2 .* (rand(N) <= p); 
A = sparse(A / max(real(eig(A))) * 0.95);   % Stabilize base matrix
% Input matrices
B = 2*(rand(N,size(XCr1,1))-.5)*sig;
C = 2*(rand(N,1)-.5)*c;
% Fixed point
r0 = (rand(N,1)*.2+.8) .* sign(rand(N,1)-0.5); % Distribution of offsets
d = atanh(r0)-A*r0;
load fig_tanh_SN_SP_mech_params.mat;
delr = @(rv,xv,cv) gam*[-rv + tanh(A*rv+B*xv+C*cv+d)];

disp('Simulating Reservoir');
RT0 = zeros(N,n); RT0(:,1) = r0;
RT1 = zeros(N,n); RT1(:,1) = r0;
RT2 = zeros(N,n); RT2(:,1) = r0;
RT3 = zeros(N,n); RT3(:,1) = r0;
Cin = ones(1,n);
fprintf([repmat('.', [1, 100]) '\n']);
nInd = 0;
for i = 2:n
    if(i > nInd*n); fprintf('='); nInd = nInd + .01; end
    RT0(:,i) = RT0(:,i-1) + delr(RT0(:,i-1),XCr0(:,i-1),0*Cin(i))*delT;
    RT1(:,i) = RT1(:,i-1) + delr(RT1(:,i-1),XCr1(:,i-1),1*Cin(i))*delT;
    RT2(:,i) = RT2(:,i-1) + delr(RT2(:,i-1),XCr2(:,i-1),2*Cin(i))*delT;
    RT3(:,i) = RT3(:,i-1) + delr(RT3(:,i-1),XCr3(:,i-1),3*Cin(i))*delT;
end
fprintf('\n');
RT = [RT0(:,ind_t) RT1(:,ind_t) RT2(:,ind_t) RT3(:,ind_t)];
XC = [XCr0(:,ind_t,1) XCr1(:,ind_t,1) XCr2(:,ind_t,1) XCr3(:,ind_t,1)];
clear RT0 RT1 RT2 RT3

% Train outputs
disp('Training W');
W = lsqminnorm(RT', XC')';      % Use least squares norm
nErr = norm(W*RT - XC);
disp(['Training error: ' num2str(nErr)]);
norm(W)


%% Predict
% Reset reservoir initial conditions
nT = 100000;
nRr = 400000;                     % # time steps to stay in place
nRs = 20000;                     % # time steps to stay in place

RPa = zeros(N,nRs);
RPb = zeros(N,nRs);
rPa = RT(:,1*n_t);
rPb = RT(:,4*n_t);

nVS = 25;
% Translate
CPaT = linspace(0,-nVS,nT);
disp('Translating')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nT
    if(i > nInd*nT); fprintf('='); nInd = nInd + .01; end
    rPa = rPa + delr(rPa,W*rPa,CPaT(i))*delT;
end
fprintf('\n');
% Rest
disp('Transient')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nRr
    if(i > nInd*nRr); fprintf('='); nInd = nInd + .01; end
    rPa = rPa + delr(rPa,W*rPa,-nVS)*delT;
end
fprintf('\n');
% Save
RPa(:,1) = rPa;
disp('Save')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nRs
    if(i > nInd*nRs); fprintf('='); nInd = nInd + .01; end
    RPa(:,i) = RPa(:,i-1) + delr(RPa(:,i-1),W*RPa(:,i-1),-nVS)*delT;
end
fprintf('\n');
XPa = W*RPa;


% Translate
CPbT = linspace(3,nVS,nT);
disp('Translating')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nT
    if(i > nInd*nT); fprintf('='); nInd = nInd + .01; end
    rPb = rPb + delr(rPb,W*rPb,CPbT(i))*delT;
end
fprintf('\n');
% Rest
disp('Transient')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nRr
    if(i > nInd*nRr); fprintf('='); nInd = nInd + .01; end
    rPb = rPb + delr(rPb,W*rPb,nVS)*delT;
end
fprintf('\n');
% Save
RPb(:,1) = rPb;
disp('Save')
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 2:nRs
    if(i > nInd*nRs); fprintf('='); nInd = nInd + .01; end
    RPb(:,i) = RPb(:,i-1) + delr(RPb(:,i-1),W*RPb(:,i-1),nVS)*delT;
end
fprintf('\n');
XPb = W*RPb;


%% Plot Training Examples and Prediction versus test trajectory
pInd = 7;
plI = (-2690:0)+size(XCrTa,2);
plI2 = (-2900:0) + size(XPa,2);

CW = winter(20);
subplot('position',subpN(pInd,:)); cla;
hold on;
plot(XCr0(1,plI),XCr0(2,plI),'linewidth',.5,'color',CL(2,:),'clipping',0);
plot(XCr1(1,plI),XCr1(2,plI),'linewidth',.5,'color',CL(3,:),'clipping',0);
plot(XCr2(1,plI),XCr2(2,plI),'linewidth',.5,'color',CL(4,:),'clipping',0);
plot(XCr3(1,plI),XCr3(2,plI),'linewidth',.5,'color',CL(5,:),'clipping',0);
plot(XPa(1,plI2),XPa(2,plI2),'-','linewidth',3,'color',CW(2,:),'clipping',0);
plot(XCrTa(1,plI),XCrTa(2,plI),'k:','linewidth',1,'color',[1 1 1]*.8,'clipping',0);
plot(XPb(1,plI2),XPb(2,plI2),'-','linewidth',3,'color',CW(end-1,:),'clipping',0);
plot(XCrTb(1,plI),XCrTb(2,plI),':','linewidth',1,'color',[1 1 1]*.2,'clipping',0);
plot([0 4]-41,[1 1]*23,':','linewidth',1,'color',[1 1 1]*.8);
plot([0 4]-41,[1 1]*19.5,':','linewidth',1,'color',[1 1 1]*.2);
plot([0 4]-41,[1 1]*16,'-','linewidth',3,'color',CW(2,:));
plot([0 4]-41,[1 1]*12.5,'-','linewidth',3,'color',CW(end-1,:));
hold off;
% Text
text(labX,subp(pInd,4)-.2,'\textbf{f}~~~~Predicting the trajectory after changing edge length',NVTitle{:});
text(-36,16,'predicted: $c = -25$',NVTextRD{:});
text(-36,12.5,'predicted: $c = 25$',NVTextRD{:});
text(-36,23,'true: length $= 39.25$',NVTextRD{:});
text(-36,19.5,'true: length $= 41.75$',NVTextRD{:});

set(gca,'visible',0);
axis([[-1 1]*40-4 [0 40]-10]);


%% Save
fName = 'fig_tanh_SN_SP_mech';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');
