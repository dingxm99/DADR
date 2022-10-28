function [ sc_lab ] = callsc( data,k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

D = L2_distance(data',data', 1);
[m,n]=size(data);
 
 [cluster_labels evd_time kmeans_time total_time] = sc(D, 5, k);
 sc_lab=cluster_labels;
%   [ sc_cen ] = DP_Cluster_Center(data,cluster_labels,k );

%  [sc_lab sc_cen]=AP_hasCen(data,k); 

end

