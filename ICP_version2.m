function [R, d, transformed_points] = icp_2d(source_points, target_points, max_iterations, tolerance)
    % ICP_2D Iterative Closest Point algorithm for 2D point clouds.
    % [R, t, transformed_points] = icp_2d(source_points, target_points, max_iterations, tolerance)
    % source_points: Nx2 matrix of source points.
    % target_points: Mx2 matrix of target points.
    % max_iterations: maximum number of iterations.
    % tolerance: convergence tolerance.
    % R: rotation matrix.
    % t: translation vector.
    % transformed_points: transformed source points.

    % Initialize transformation
    R = eye(2);
    d = zeros(2, 1);
    
    % Loop until convergence or maximum iterations
    for iter = 1:max_iterations
        % Step 1: Find the closest points
        indices = knnsearch(target_points, source_points);
        closest_points = target_points(indices, :);
        
        indices
        % Step 2: Compute the mean of the source and closest points
        mean_source = mean(source_points);
        mean_closest = mean(closest_points);
        
        % Step 3: Subtract the means
        source_centered = source_points - mean_source;
        closest_centered = closest_points - mean_closest;
        
        % Step 4: Compute the covariance matrix
        H = source_centered' * closest_centered;
        
        % Step 5: Singular Value Decomposition (SVD)
        [U, ~, V] = svd(H);
        
        % Step 6: Compute the rotation matrix
        R_new = V * U';
        
        % Ensure a proper rotation (det(R) == 1)
        if det(R_new) < 0
            V(:,end) = -V(:,end);
            R_new = V * U';
        end
        
        % Step 7: Compute the translation vector
        d_new = mean_closest' - R_new * mean_source';
        
        % Transform the source points
        transformed_points = (R_new * source_points' + d_new)';
        
        % Check for convergence
        if norm(R_new - R, 'fro') < tolerance && norm(d_new - d) < tolerance
            break;
        end
        % Update transformation
        R = R_new;
        d = d_new;
    end
    
    % Apply final transformation
    transformed_points = (R * source_points' + d)';
end