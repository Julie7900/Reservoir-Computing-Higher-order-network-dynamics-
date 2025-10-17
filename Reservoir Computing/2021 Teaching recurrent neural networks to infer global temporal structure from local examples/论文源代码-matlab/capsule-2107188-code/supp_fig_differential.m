%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(7); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 9.3];                       % Figure Size in cm  [w,h]
fMarg = [.4 .1 .3 .2];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.00 4.75 4.75 4.75];...      % Subplot position in cm [x,y,w,h]
        [ 0.00 0.00 4.75 4.75];...
        [ 4.75 4.75 4.75 4.75];...
        [ 4.75 0.00 4.75 4.75];...
        [ 9.50 4.75 4.75 4.75];...
        [ 9.50 0.00 4.75 4.75];...
        [14.25 4.75 4.75 4.75];...
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
CO = [[150 100 100];...
      [170 100 100];...
      [190 100 100]]/255;
CP = [[200 180 120];...
      [220 180 100];...
      [230 210 120]]/255;


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
M = 3;                                      % Number of Lorenz states
gam = 100;                                  % Reservoir responsiveness
sig = 0.008;                                % Attractor influence
c = .004;                                   % Control Parameter
p = 0.1;                                    % Reservoir initial density
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
load supp_fig_differential_params.mat;
R2 = ReservoirTanh(A,B,C, r0,x0,c0, delT, gam);   % Reservoir system
L0 = Lorenz(Lx0, delT, [10 28 8/3]);            % Lorenz system
     

%% Lorenz time series
disp('Simulating Attractor');
X0 = L0.propagate(n);                           % Generate time series


%% Translation
Cin = ones(1,n,4);
CT = [0*Cin 1*Cin 2*Cin 3*Cin];

a = [1 0 0]';
XT = [X0 X0+a X0+2*a X0+3*a];

% Drive reservoir
R2.B = B;
disp('Simulating Reservoir');
RT = R2.train(XT,CT);
RT = RT(:,t_ind);
% Train outputs
disp('Training WT');
WT = lsqminnorm(RT', XT(:,t_ind,1)')';        % Use least squares norm
disp(['Training error: ' num2str(norm(WT*RT - XT(:,t_ind,1)))]);


%% Prediction
R2.r = RT(:,n_t);
RTP = R2.predict_x(zeros(1,20000,4),WT);


%% Reservoir states and predicted changes: Translation
nPT = 1:10:15000; lnPT = length(nPT);
delRT0 = zeros(N,lnPT);
delRT1 = zeros(N,lnPT);
delRT2 = zeros(N,lnPT);
delRT3 = zeros(N,lnPT);

MTS = A + B*WT;
IN = eye(N);

disp('Computing predicted change in states: translation');
RTS = RTP(:,nPT);

T = tanh(MTS*RTS+R2.d);
K = (1-T.^2);
dRTS = gam*(-RTS + T);

Tdot = K.*(MTS*dRTS);
Kdot = -2*T.*Tdot;
ddRTS = gam*(-dRTS + Tdot);
    
Tddot = K.*(MTS*ddRTS) + Kdot.*(MTS*dRTS);
Kddot = -2*(Tdot.^2 + T.*Tddot);
dddRTS = gam*(-ddRTS + Tddot);
    
Tdddot = K.*(MTS*dddRTS) + 2*Kdot.*(MTS*ddRTS) + Kddot.*(MTS*dRTS);
Kdddot = -2*(3*Tdot.*Tddot + T.*Tdddot);

% Order zero approximation
fprintf([repmat('.', [1, 100]) '\n']); nInd = 0;
for i = 1:lnPT
    if(i > nInd*lnPT); fprintf('='); nInd = nInd + .01; end
    % 0th order
    SD = IN-K(:,i).*A;
    U = K(:,i).*(C+B*a);
    delRT0(:,i) = 20*SD\U;
    
    % 1st order
    SAI = (IN/gam)^-1;
    SB =  IN-K(:,i).*A;
    SC =  IN-K(:,i).*A;
    SD =  -Kdot(:,i).*A;
    S = -(SC*SAI*SB - SD)\[-SC*SAI IN];
    U = [K(:,i).*(C+B*a); Kdot(:,i).*(C+B*a)];
    delRT1(:,i) = 20*S*U;
    
    % 2nd order
    SAI = [ gam*IN            zeros(N);...
           -gam^2*(IN-K(:,i).*A)   gam*IN];
    SB =  [IN-K(:,i).*A;...
           -Kdot(:,i).*A];
    SC =  [-2*Kdot(:,i).*A  IN-K(:,i).*A];
    SD =   -Kddot(:,i).*A;
    S = -(SC*SAI*SB - SD)\[-SC*SAI IN];
    U = [K(:,i).*(C+B*a); Kdot(:,i).*(C+B*a); Kddot(:,i).*(C+B*a)];
    delRT2(:,i) = 20*S*U;
    
    % 3rd order
    SAI = [ gam*IN                                zeros(N)     zeros(N);...
           -gam^2*(IN-K(:,i).*A)                       gam*IN       zeros(N);...
            gam^3*(IN-K(:,i).*A)^2 + 2*gam^2*Kdot(:,i).*A  -gam^2*(IN-K(:,i).*A)   gam*IN];
    SB =  [IN-K(:,i).*A;...
           -Kdot(:,i).*A;...
           -Kddot(:,i).*A];
    SC =  [-3*Kddot(:,i).*A  -3*Kdot(:,i).*A  IN-K(:,i).*A];
    SD =   -Kdddot(:,i).*A;
    S = -(SC*SAI*SB - SD)\[-SC*SAI IN];
    U = [K(:,i).*(C+B*a); Kdot(:,i).*(C+B*a); Kddot(:,i).*(C+B*a); Kdddot(:,i).*(C+B*a)];
    delRT3(:,i) = 20*S*U;
end
fprintf('\n');


%% Plot Reservoirs
% subsample
[~,nSrtT] = sort(sum((delRT1 - mean(delRT1,2)).^2,2));
nIndT = (N-3:N);
nSh = ([1 2.2 3.4 4.6] + 0)/10;

% Plot
pInd = 1; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndT)
    nP = nSrtT(nIndT(i));
    RTsi = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))],.002);
    RTsi2 = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))+delRT0(nP,:)],.002);
    plot(RTsi(1,:),RTsi(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RTsi2(1,:),RTsi2(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);

% Text
text(labX,subp(pInd,4)-.2,'\textbf{a}\hspace{.2cm} 0th order approximation',NVTitle{:});
text(.5,-.025,'time',NVTextH{:});
text(labX,3.3,'$r_1$',NVTitle{:});
text(labX,2.4,'$r_2$',NVTitle{:});
text(labX,1.5,'$r_3$',NVTitle{:});
text(labX,0.6,'$r_4$',NVTitle{:});


% Plot
pInd = 3; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndT)
    nP = nSrtT(nIndT(i));
    RTsi = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))],.002);
    RTsi2 = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))+delRT1(nP,:)],.002);
    plot(RTsi(1,:),RTsi(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RTsi2(1,:),RTsi2(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);
text(labX,subp(pInd,4)-.2,'\textbf{b}\hspace{.2cm} 1st order approximation',NVTitle{:});


% Plot
pInd = 5; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndT)
    nP = nSrtT(nIndT(i));
    RTsi = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))],.002);
    RTsi2 = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))+delRT2(nP,:)],.002);
    plot(RTsi(1,:),RTsi(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RTsi2(1,:),RTsi2(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);
text(labX,subp(pInd,4)-.2,'\textbf{c}\hspace{.2cm} 2nd order approximation',NVTitle{:});


% Plot
pInd = 7; 
subplot('position',subpN(pInd,:)); cla;
set(gca,'visible',0);
hold on;
for i = 1:length(nIndT)
    nP = nSrtT(nIndT(i));
    RTsi = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))],.002);
    RTsi2 = downsample_curvature([nPT;RTS(nP,:)-mean(RTS(nP,:))+delRT3(nP,:)],.002);
    plot(RTsi(1,:),RTsi(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(1,:));
    plot(RTsi2(1,:),RTsi2(2,:)*.2+nSh(i),'-','linewidth',.7,'color',CP(3,:));
end
plot_spline([min(nPT) max(nPT); [1 1]*.05],'linewidth',1,'head',1,'headpos',1);
hold off;
axis([min(nPT) max(nPT) [0 .55]+.03]);
text(labX,subp(pInd,4)-.2,'\textbf{d}\hspace{.2cm} 3rd order approximation',NVTitle{:});

% Text
% text(labX,subp(pInd,4)-.2,'\textbf{b}\hspace{1.9cm}reservoir trained on translated input',NVTitle{:});


%% 3D View
XTP = WT*RTS;
XTPT0 = WT*(RTS + delRT0);
XTPT1 = WT*(RTS + delRT1);
XTPT2 = WT*(RTS + delRT2);
XTPT3 = WT*(RTS + delRT3);

CW = winter(9);

pInd = 2; 
subplot('position',subpN(pInd,:)); cla;
XTPds = downsample_curvature(XTP,.4,[155,15]) - [0;0;23];
XTPTds = downsample_curvature(XTPT0,.4,[155,15]) - [0;0;23];
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


pInd = 4; 
subplot('position',subpN(pInd,:)); cla;
XTPds = downsample_curvature(XTP,.4,[155,15]) - [0;0;23];
XTPTds = downsample_curvature(XTPT1,.4,[155,15]) - [0;0;23];
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
drawnow;


pInd = 6; 
subplot('position',subpN(pInd,:)); cla;
XTPds = downsample_curvature(XTP,.4,[155,15]) - [0;0;23];
XTPTds = downsample_curvature(XTPT2,.4,[155,15]) - [0;0;23];
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
drawnow;


pInd = 8; 
subplot('position',subpN(pInd,:)); cla;
XTPds = downsample_curvature(XTP,.4,[155,15]) - [0;0;23];
XTPTds = downsample_curvature(XTPT3,.4,[155,15]) - [0;0;23];
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
drawnow;



%% Save
fName = 'supp_fig_differential.pdf';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');