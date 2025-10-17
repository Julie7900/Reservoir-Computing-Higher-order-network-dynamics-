%% Prepare Space
clear; clc;
set(groot,'defaulttextinterpreter','latex');


%% Parameters and Dimensions
fig = figure(8); clf;
delete(findall(gcf,'type','annotation'));

% Figure parameters
FS = 10;                                % Fontsize
fSize = [19 10.0];                      % Figure Size in cm  [w,h]
fMarg = [.5 .5 .6 .5];                  % Margins in cm, [l,r,d,u]
subp = [[ 0.60 5.35 4.75  4.75];...     % Subplot position in cm [x,y,w,h]
        [ 5.20 5.35 4.75  4.75];...
        [ 9.80 5.35 4.75  4.75];...
        [14.40 5.35 4.75  4.75];...
        [ 0.60 0.30 4.75  4.75];...
        [ 5.20 0.30 4.75  4.75];...
        [ 9.80 0.30 4.75  4.75];...
        [14.45 0.30 4.75  4.75]];

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

  
%% Load Data
load dist_tanh_450_trs.mat;
load dist_tanh_450_trf.mat;
load dist_tanh_450_bif.mat;
load dist_tanh_450_bif2.mat;
load dist_wc_600_trs.mat;
load dist_wc_600_trf.mat;
load dist_wc_600_bif.mat;
load dist_wc_600_bif2.mat;


%% Plot Data
bw = 0.15;
% Attractor Distances
Dall = cell(8,1);
Dall{1} = Dtanhtrs;
Dall{2} = Dtanhtrf;
Dall{3} = Dtanhbif;
Dall{4} = Dtanhbif2;
Dall{5} = Dwctrs;
Dall{6} = Dwctrf;
Dall{7} = Dwcbif;
Dall{8} = Dwcbif2;
% Time step distances
DCompall = cell(8,1);
DCompall{1} = DtanhComptrs;
DCompall{2} = DtanhComptrf;
DCompall{3} = DtanhCompbif;
DCompall{4} = DtanhCompbif2;
DCompall{5} = DwcComptrs;
DCompall{6} = DwcComptrf;
DCompall{7} = DwcCompbif;
DCompall{8} = DwcCompbif2;

sTitle = {'~~~~~~tanh translate','~~~~~~tanh transform','~~~~~~tanh bifurcate 1','~~~~~~tanh bifurcate 2',...
          'Wilson-Cowan translate','Wilson-Cowan transform','Wilson-Cowan bifurcate 1','Wilson-Cowan bifurcate 2'};
sCapt = {'a','b','c','d','e','f','g','h'};

set(gcf,'color','w');
for i = 1:8
    subplot('position',subpN(i,:)); cla;
    hold on;
    histogram(log10(sqrt(Dall{i})),'binwidth',bw,'normalization','probability','linewidth',.3);
    histogram(log10(sqrt(DCompall{i})),'binwidth',bw,'normalization','probability','linewidth',.3);
    hold off;
    axis([-3.5 0.5 0 0.45]);
    text(labX,subp(i,4)+.2,['\textbf{' sCapt{i} '} ' sTitle{i}],NVTitle{:});
    set(gca,'TickLabelInterpreter','latex','FontSize',FS);
    if(i==1)
        legend('error','step size',NVTextR{:},'interpreter','latex');
    end
    if(i==1 || i==5)
        text(-.23,.35,'frequency',NVTextR{:},'rotation',90);
    end
    if(i>=5)
        text(0.5,-0.19,'$\log_{10}(\mathrm{distance})$',NVTextH{:});
    end
    drawnow;
end


%% Save
fName = 'supp_fig_attractor_similarity';
set(gcf, 'Renderer', 'painters'); 
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 fSize];
fig.PaperSize = fSize;
saveas(fig, ['..\results\' fName], 'pdf');
set(gcf, 'Renderer', 'opengl');