function [XC, dInd] = downsample_curvature(X,a,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for downsampling the curvature of data.
%
% Inputs:
%
%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(nargin == 3)
    XS = X;
    % Get view vector
    vx = [ sind(v(1))*cosd(v(2));...
          -cosd(v(1))*cosd(v(2));...
           sind(v(2))];
    vyz = null(vx');
    % Project time series onto 2D plane
    X = vyz'*X;
end

VInd = [1 1];
n_start = size(X,2);
aInd = 1:n_start;
dInd = [];

while(length(VInd) > 1)
    % Initialize
    V = diff(X,1,2);
    VMag = sum(V(:,2:end) .* V(:,1:end-1)) ./ sqrt(sum(V(:,1:end-1).^2));
    VNorm = [a+1 sqrt(sum(V(:,2:end).^2) - VMag.^2)];
    
    % Find indices where curvature is already greater than a
    VIndL = find(VNorm < a);
    VIndU = find(VNorm > a);
    
    % Remove indices around points that are already too sharp
    VIndL = setdiff(VIndL,[VIndU+1 VIndU-1]);
    if(isempty(VIndL)); break; end
    
    % Search for consecutive points
    VIndD = [2 diff(VIndL)];
    VIndNC = VIndL(VIndD ~= 1);          % Nonconsecutive indices
    VIndC = VIndL(VIndD == 1);           % Consecutive indices
    
    % Putative points to remove
    VIndR = [VIndNC VIndC(1:2:end)];    % Remove every other conescutive
    % Putative points to keep
    VIndK = setdiff(1:size(X,2),VIndR);
    
    % Curvature of points to keep
    XP = X(:,VIndK);
    V = diff(XP,1,2);
    VMag = sum(V(:,2:end) .* V(:,1:end-1)) ./ sqrt(sum(V(:,1:end-1).^2));
    VNorm = [a sqrt(sum(V(:,2:end).^2) - VMag.^2)];
    VIndKO = VIndK(VNorm > a);
    
    % Remove bad points to remove
    VIndR = setdiff(VIndR,[VIndKO+1 VIndKO-1]);
    if(isempty(VIndR)); break; end
    
    % Keep track of removed points
    dInd = [dInd aInd(VIndR)];          % Collected removed indices
    dr = setdiff(1:size(X,2),VIndR);
    aInd = aInd(dr);                    % Remaining indices
    X = X(:,dr);
end
dInd = setdiff(1:n_start,dInd);
if(nargin == 3)
    X = XS(:,dInd);
end
XC = X(:,1:end);
disp(['compression ratio: ' num2str(size(XC,2)/n_start)]);
end