
close all;

lidat_points_T0 = [1 1; 2 2; 4 4; 3 3; 5 5];
%source_points_1 = [1 1; 2 2; 4 4; 3 3; 5 5; 6 6];
max_iterations = 100;
tolerance = 1e-8;
angle_rad_theorique = deg2rad(40);
angle_rad_mesure = deg2rad(40+ random('Normal', 0, 10));
rotation_matrix_2D_theorique = [cos(angle_rad_theorique), -sin(angle_rad_theorique); sin(angle_rad_theorique), cos(angle_rad_theorique)];
rotation_matrix_2D_mesuree = [cos(angle_rad_mesure), -sin(angle_rad_mesure); sin(angle_rad_mesure), cos(angle_rad_mesure)];
t_theorique = [2;2];
t_mesure = [t_theorique(1)+ random('Normal', 0, 0.5); t_theorique(2)+ random('Normal', 0, 0.5)];

lidat_points_T1_Theorique = (rotation_matrix_2D_theorique * lidat_points_T0' + t_theorique)';
lidat_points_T1_Mesure = (rotation_matrix_2D_mesuree * lidat_points_T0' + t_mesure)';

[lidat_t0_matching, lidat_t1_matching] = matching(lidat_points_T0,lidat_points_T1_Mesure);

[R1, t1, lidat_points_T1_recale1] = icp_2d(lidat_t0_matching, lidat_t1_matching, max_iterations, tolerance); 
[R, t, lidat_points_T1_recale] = icp_2d(lidat_points_T0, lidat_points_T1_Mesure, max_iterations, tolerance);
figure;
hold on;
scatter(lidat_points_T0(:,1), lidat_points_T0(:,2), 'k');
scatter(lidat_points_T1_Theorique(:,1), lidat_points_T1_Theorique(:,2), 'r*');
%scatter(transformed_points(:,1), transformed_points(:,2), 'gx');
scatter(lidat_points_T1_Mesure(:,1), lidat_points_T1_Mesure(:,2), 'bo');
scatter(lidat_points_T1_recale1(:,1), lidat_points_T1_recale1(:,2), 'gx');

xlim([0 10]);
ylim([0 10]);
legend('Lidar Pts t0','Lidar Pts Theorique', 'Lidar Pts Mesure', 'Lidar Pts Recale');