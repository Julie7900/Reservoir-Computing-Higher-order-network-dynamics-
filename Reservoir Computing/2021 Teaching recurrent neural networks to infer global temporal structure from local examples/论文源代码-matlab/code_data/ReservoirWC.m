classdef ReservoirWC < handle
    properties
        % Matrices
        A               % N x N matrix of internal reservoir connections
        B               % N x M matrix of s dynamical inputs to learn
        C               % N x K matrix of k external inputs for control
        R               % N x N matrix of autonomous reservoir connections
        W               % K x N matrix of trained output weights
        re              % 1 x 1 scalar of refractory period
        % States and fixed points
        r               % N x 1 vector of current state
        rs              % N x 1 vector of reservoir fixed point
        xs              % s x 1 vector of input fixed point
        cs              % k x 1 vector of control fixed point
        d               % N x 1 vector of bias terms
        % Time
        delT            % Timescale of simulation
        gam             % Gamma: Timescale of reservoir evolution speed
    end
    
    methods
        % Constructor
        function obj = ReservoirWC(A, B, C, rs, xs, cs, delT, gam)
            % Matrices
            obj.A = A;
            obj.B = B;
            obj.C = C;
            obj.re = 0.0;
            % States and fixed points
            obj.rs = rs;
            obj.xs = xs;
            obj.cs = cs;
            obj.d = -A*rs - B*xs - C*cs + log(rs ./ (1 - (1+obj.re).*rs));
            % Time
            obj.delT = delT;
            obj.gam = gam;
            % Initialize reservoir states to 0
            obj.r = zeros(size(A,1),1);
        end
        
        % Training: input both inputs x and control c 
        function D = train(o, x, c)
            nInd = 0; nx = size(x,2);                 % Counter
            D = zeros(size(o.A,1), nx);
            D(:,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nx
                if(i > nInd*nx); fprintf('='); nInd = nInd + .01; end
                o.propagate(x(:,i-1,:),c(:,i-1,:)); % Propagate States
                D(:,i) = o.r;
            end
            fprintf('\n');
        end
        
        % Prediction: input only control c
        function D = predict_x(o, c, W)
            nInd = 0; nc = size(c,2);               % Counter
            o.R = o.A + o.B*W;                      % Feedback
            o.W = W;
            D = zeros(size(o.R,1), nc);
            D(:,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nc
                if(i > nInd*nc); fprintf('='); nInd = nInd + .01; end
                o.propagate_x(c(:,i-1,:));            % Propagate States
                D(:,i) = o.r;
            end
            fprintf('\n');
        end
        
        
        %% RK4 integrator
        % driven reservoir
        function propagate(o,x,c)
            k1 = o.delT * o.del_r(o.r,        x(:,1,1), c(:,1,1));
            k2 = o.delT * o.del_r(o.r + k1/2, x(:,1,2), c(:,1,2));
            k3 = o.delT * o.del_r(o.r + k2/2, x(:,1,3), c(:,1,3));
            k4 = o.delT * o.del_r(o.r + k3,   x(:,1,4), c(:,1,4));
            o.r = o.r + (k1 + 2*k2 + 2*k3 + k4)/6;
        end
        % feedback reservoir
        function propagate_x(o,c)
            k1 = o.delT * o.del_r_x(o.r,        c(:,1,1));
            k2 = o.delT * o.del_r_x(o.r + k1/2, c(:,1,2));
            k3 = o.delT * o.del_r_x(o.r + k2/2, c(:,1,3));
            k4 = o.delT * o.del_r_x(o.r + k3,   c(:,1,4));
            o.r = o.r + (k1 + 2*k2 + 2*k3 + k4)/6;
        end
        
        
        %% ODEs
        % driven reservoir
        function dr = del_r(o,r,x,c)
            dr = o.gam * (-r + (1-o.re*r)./(1+exp(-o.A*r - o.B*x - o.C*c - o.d)));
        end
        % feedback reservoir
        function dr = del_r_x(o,r,c)
            dr = o.gam * (-r + (1-o.re*r)./(1+exp(-o.A*r - o.B*(o.W*r) - o.C*c - o.d)));
        end
    end
end