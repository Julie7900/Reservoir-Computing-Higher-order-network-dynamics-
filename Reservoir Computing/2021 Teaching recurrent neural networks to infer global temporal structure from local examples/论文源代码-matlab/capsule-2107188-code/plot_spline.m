function [] = plot_spline(X,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for plotting spline lines
%
% Inputs
% X:        2 x n       matrix of spline points
%
% Outputs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Parser
p = inputParser;
addParameter(p,'color',[0 0 0]);
addParameter(p,'linewidth',.7);
addParameter(p,'headwidth',5);
addParameter(p,'headlength',5);
addParameter(p,'headpos',.5);
addParameter(p,'head',0);
addParameter(p,'ratio',1);
addParameter(p,'prntslp',0);
parse(p,varargin{:});

CL = p.Results.color;
LW = p.Results.linewidth;
HW = p.Results.headwidth;
HL = p.Results.headlength;
HP = p.Results.headpos;
h  = p.Results.head;
r  = p.Results.ratio;
PS = p.Results.prntslp;

%% Plot
% Line
coeffs = cscvn(X);
[pt,t] = fnplt(coeffs);
plot(pt(1,:),pt(2,:),'-','linewidth',LW,'color',CL);

% Print slope
if(PS ~= 0)
    disp([(pt(2,2)-pt(2,1))/(pt(1,2)-pt(1,1)),...
          (pt(2,end)-pt(2,end-1))/(pt(1,end)-pt(1,end-1))]);
end

% Arrow
if(h ~= 0)
    pInd = floor(HP*(length(t)-2)+2);
    ah = annotation('arrow','HeadLength',HL,'HeadWidth',HW,'color',CL);
    set(ah,'parent',gca,'position',...
        [pt(1,pInd) pt(2,pInd) (pt(1,pInd)-pt(1,pInd-1))/1000 (pt(2,pInd)-pt(2,pInd-1))/1000*r]);
end




end
