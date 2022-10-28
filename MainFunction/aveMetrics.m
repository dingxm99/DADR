function [output_ave] = aveMetrics(input_data)
    ind = input_data>0;
    count = sum(ind,3);
    sum_input_data = sum(input_data,3);
    output_ave = sum_input_data ./ count;
end

