function output_best_ave = aveBestMetrics(input_data,time)
    input_data(:,1:2,:)=[];
	[max_input_data, max_index] = max(input_data,[],2);
    output_best_ave = sum(max_input_data,3)/time;
end
