function [] = visualize_network(Xs, Xu, conn, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is to visualize a constructed network.
%
% Inputs
% Xs:       d x n  matrix of specified node positions
% Xu:       d x m  matrix of unspecified node positions
% conn:     k x 2  matrix of edges from node i to node j
%
% Name/Value Pairs
% scale     1 x 1       size: network node and edge     [1]
% nalpha    1 x 1       transparency: node              [1]
% lalpha    1 x 1       transparency: edge              [1]
% scolor    1 x 3       color: specified node           [255 100 100]/255
% ucolor    1 x 3       color: unspecified node         [100 100 255]/255
% bcolor    1 x 3       color: node boarder             [0 0 0]
% lcolor    1(k) x 3    color: edge                     [0 0 0]
% lwidth    1 x 1       width: edge                     [1.5]
% bwidth    1 x 1       width: node boarder             [0.6]
% msize     1 x 1       size: nodes                     [4]
%
% Outputs
% Figure. Assumes the desired figure placement is already selected
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Parse Inputs
p = inputParser;
addParameter(p,'scale',1);
addParameter(p,'nalpha',1);
addParameter(p,'lalpha',1);
addParameter(p,'scolor',repmat([230 225 225]/255,max(size(Xs,2),1),1));
addParameter(p,'ucolor',repmat([225 225 230]/255,max(size(Xu,2),1),1));
addParameter(p,'bcolor',[1 1 1]*.0);
addParameter(p,'lcolor',repmat([0 0 0],size(conn,1),1));
addParameter(p,'lwidth',.5);
addParameter(p,'bwidth',.5);
addParameter(p,'msize',4);
parse(p,varargin{:});

sC = p.Results.scale;
nTr = p.Results.nalpha;
lTr = p.Results.lalpha;
C_SN = p.Results.scolor;
C_UN = p.Results.ucolor;
C_L = p.Results.lcolor;
C_B = p.Results.bcolor;
lw = p.Results.lwidth*sC;
ms = p.Results.msize*sC;
bw = p.Results.bwidth*sC;


% Artifically implement alpha transparency for nodes; no native support
% if(size(C_SN,2) == 3)
%     C_SN = (C_SN-1)*nTr + 1;
% end
% if(size(C_UN,2) == 3)
%     C_UN = (C_UN-1)*nTr + 1;
% end
% C_B = (C_B - 1)*nTr + 1;

% Replicate colors if size of color list is less than number of nodes
C_SN = repmat(C_SN,size(Xs,2)/size(C_SN,1),1);
C_UN = repmat(C_UN,size(Xu,2)/size(C_UN,1),1);
nRep = ceil(size(conn,1)/size(C_L,1));
C_L = reshape(permute(repmat(C_L,1,1,nRep),[2,3,1]),3,size(C_L,1)*nRep)';
C_L(:,4) = lTr;


%% Plot Parameters
d = size(Xs,1);                     % Number of Dimensions

hold on
X = [Xs Xu];
if(d==2)
    % Edges
    for i = 1:size(conn,1)
        line([X(1,conn(i,1)); X(1,conn(i,2))],...
             [X(2,conn(i,1)); X(2,conn(i,2))],...
             'linewidth', lw, 'color', C_L(i,:),'clipping',0);
    end
    % Specified Nodes
    if(size(Xs,2) > 0)
        scatter(Xs(1,:),Xs(2,:),ms^2,C_SN(1:size(Xs,2),:),'filled',...
                'markeredgecolor',C_B,'linewidth',bw,'clipping',0,...
                'markerfacealpha',nTr,'markeredgealpha',lTr);
%         for i = 1:size(Xs,2)
%             plot(Xs(1,:),Xs(2,:),'o','markerfacecolor',C_SN(i,:),...
%                  'markeredgecolor',C_B,'linewidth',bw,'clipping',0,...
%                  'markersize',ms,'markeralpha',nTr);
%         end
    end
    % Unspecified Nodes
    if(size(Xu,2) > 0)
        scatter(Xu(1,:),Xu(2,:),ms^2,C_UN(1:size(Xu,2),:),'filled',...
                'markeredgecolor',C_B,'linewidth',bw,'clipping',0,...
                'markerfacealpha',nTr,'markeredgealpha',lTr);
    end
    set(gca,'visible',0);
    set(gcf,'color','w');
elseif(d==3)
    % Spherical point
    [xSp, ySp, zSp] = sphere(20);
    xSp = xSp/10*sC; 
    ySp = ySp/10*sC; 
    zSp = zSp/10*sC; 
    % Edges
    for i = 1:size(conn,1)
        line([X(1,conn(i,1)); X(1,conn(i,2))],...
             [X(2,conn(i,1)); X(2,conn(i,2))],...
             [X(3,conn(i,1)); X(3,conn(i,2))],...
             'linewidth', lw, 'color', C_L(i,:));
    end
    % Unspecified Nodes
    for i = 1:size(Xu,2)
        s = surf(xSp+Xu(1,i), ySp+Xu(2,i), zSp+Xu(3,i));
        s.FaceColor = C_UN(i,:);
        s.EdgeColor = 'none';
    end
    % Specified Nodes
    for i = 1:size(Xs,2)
        s = surf(xSp+Xs(1,i), ySp+Xs(2,i), zSp+Xs(3,i));
        s.FaceColor = C_SN(i,:);
        s.EdgeColor = 'none';
    end
    hold off;
    set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[],...
            'XTick',[],'YTick',[],'ZTick',[],'box','on','boxstyle','back');
end
end