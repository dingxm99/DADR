
clc;clear;

filename=['D014.mat';'D017.mat';'D030.mat';'D040.mat';'D043.mat';
        'D047.mat';'D143.mat';'D151.mat';'D207.mat'];

timecount = 20;
maxlayer = 14;
var_coefficient = 0.09;
allFilecount = size(filename,1);
% Initialization
all_dims_1 = zeros(allFilecount,maxlayer+2,timecount);
all_accuracy_sc_1 = zeros(allFilecount,maxlayer+2,timecount);
rand_sc_1 = zeros(allFilecount,maxlayer+2,timecount);
ar_sc_1 = zeros(allFilecount,maxlayer+2,timecount);
jac_sc_1 = zeros(allFilecount,maxlayer+2,timecount);
fm_sc_1 = zeros(allFilecount,maxlayer+2,timecount);
hiddata = cell(allFilecount,timecount);

for time=1:timecount
    fprintf('No %d times \n',time); 
    for loop = 1:allFilecount
        all_dims_1(loop,1,time)=loop;
        %% Import data
        file = filename(loop,:);
        fprintf('file =%s \n',file);
        if exist('lables')
            clear lables;
        end
        if exist('data')
            clear data;
        end
        load(file);
        
        [cases, dims]=size(data);
        if(cases>5000 || dims>2000)
            all_dims_1(loop,1,time)=0;
            continue;
        end
        all_dims_1(loop,2,time)=dims;
        
         %% Test data with SC clustering
        if exist('lables')
            labels=lables;
        end
        if exist('label')
            labels=label;
        end
        k = max(labels)-min(labels)+1;
        if cases<=5000 && dims<=2000 && cases>0 && size(labels,2)==1
            try    
                fprintf('start SC....\n');
                [datalabels] = callsc(data,k); 
                sc_org_score = accuracy(labels, datalabels);
                all_accuracy_sc_1(loop,1,time) = sc_org_score/100;
                other_metrics = valid_external(labels, datalabels);
                rand_sc_1(loop,1,time) = other_metrics(1);
                ar_sc_1(loop,1,time) = other_metrics(2);
                jac_sc_1(loop,1,time) = other_metrics(3);
                fm_sc_1(loop,1,time) = other_metrics(4);
            catch
                all_accuracy_sc_1(loop,1,time) = -1;
                rand_sc_1(loop,1,time) = -1;
                ar_sc_1(loop,1,time) = -1;
                jac_sc_1(loop,1,time) = -1;
                fm_sc_1(loop,1,time) = -1;
                datalabels = zeros(size(data)); 
            end
        end     
        
        hidcode = rbm_bc_linedecoder(data,dims);
        hiddcode = hidcode;
        
        %% Test hidcode with SC clustering
        k = max(labels)-min(labels)+1;
        if cases<=5000 && dims<=2000 && cases>0 && size(labels,2)==1
            try    
                fprintf('start SC....\n');
                [datalabels] = callsc(hidcode,k); 
                sc_org_score = accuracy(labels, datalabels);
                all_accuracy_sc_1(loop,2,time) = sc_org_score/100; 
                other_metrics = valid_external(labels, datalabels);
                rand_sc_1(loop,2,time) = other_metrics(1);
                ar_sc_1(loop,2,time) = other_metrics(2);
                jac_sc_1(loop,2,time) = other_metrics(3);
                fm_sc_1(loop,2,time) = other_metrics(4);
            catch
                all_accuracy_sc_1(loop,2,time) = -1;
                rand_sc_1(loop,2,time) = -1;
                ar_sc_1(loop,2,time) = -1;
                jac_sc_1(loop,2,time) = -1;
                fm_sc_1(loop,2,time) = -1;
                datalabels = zeros(size(data)); 
            end
        end     
        
            
        %% Deep adaptive dimensionality reduction
        for numlayer=1:maxlayer %numlayer²ãÊý
            
            [hid_cases, hid_dims] = size(hidcode);
            stdcon = zeros(1,hid_dims);
            for i=1:hid_dims
                stdcon(i) = std(hidcode(:,i)) / mean(hidcode(:,i)); 
            end
            optcol = find(stdcon > var_coefficient);
            opt_dims = length(optcol);
            opt_hidcodes = zeros(hid_cases,opt_dims);
            t=1;
            for j=1:size(optcol,2)
                opt_hidcodes(:,t)=hidcode(:,optcol(1,j));
                t=t+1;
            end 

            [opt_cases, opt_dims]=size(opt_hidcodes);
            if(opt_dims==0)
                hiddata{loop,time} = hidcode;
                break;
            end
            if(opt_dims==hid_dims)
                hiddata{loop,time} = hidcode;
                numlayer=numlayer-1;
                break;
            end
            all_dims_1(loop,numlayer+2,time)=opt_dims;
            
           
            hiddatas = rbm_bc_linedecoder(opt_hidcodes,opt_dims);
            hidcode = hiddatas;
            
           %% SC clustering
            if exist('lables')
                labels=lables;
            end
            if exist('label')
                labels=label;
            end
            k = max(labels)-min(labels)+1;
            if cases<=5000 && dims<=2000 && cases>0 && size(labels,2)==1
                try    
                    fprintf('start DP....\n');
                    [datalabels] = callsc(hiddatas,k); 
                    sc_org_score = accuracy(labels, datalabels);
                    all_accuracy_sc_1(loop,numlayer+2,time) = sc_org_score/100;
                    other_metrics = valid_external(labels, datalabels);
                    rand_sc_1(loop,numlayer+2,time) = other_metrics(1);
                    ar_sc_1(loop,numlayer+2,time) = other_metrics(2);
                    jac_sc_1(loop,numlayer+2,time) = other_metrics(3);
                    fm_sc_1(loop,numlayer+2,time) = other_metrics(4);
                catch
                    all_accuracy_sc_1(loop,numlayer+2,time) = -1;
                    rand_sc_1(loop,numlayer+2,time) = -1;
                    ar_sc_1(loop,numlayer+2,time) = -1;
                    jac_sc_1(loop,numlayer+2,time) = -1;
                    fm_sc_1(loop,numlayer+2,time) = -1;
                    datalabels = zeros(size(hiddatas)); 
                end
            end
              
        end
        
    end
    
end

%% Calculating the average evaluation index of each layer
 cache_all_accuracy_sc_1 = all_accuracy_sc_1;
 ave_accuracy_sc_1 = aveMetrics(cache_all_accuracy_sc_1);
 ave_rand_sc_1 = aveMetrics(rand_sc_1);
 ave_ar_sc_1 = aveMetrics(ar_sc_1);
 ave_jac_sc_1 = aveMetrics(jac_sc_1);
 ave_fm_sc_1 = aveMetrics(fm_sc_1);
 ave_dim_sc_1 = aveMetrics(all_dims_1);

%% calculate the mean value of the best evaluation index corresponding to each dataset
 best_accuracy_sc_1 = aveBestMetrics(cache_all_accuracy_sc_1,timecount);
 best_rand_sc_1 = aveBestMetrics(rand_sc_1,timecount);
 best_ar_sc_1 = aveBestMetrics(ar_sc_1,timecount);
 best_jac_sc_1 = aveBestMetrics(jac_sc_1,timecount);
 best_fm_sc_1 = aveBestMetrics(fm_sc_1,timecount);

 %% save data  
save('sc_DADR.mat','filename','hiddata','all_dims_1','ave_dim_sc_1','all_accuracy_sc_1', ...
    'ave_accuracy_sc_1','best_accuracy_sc_1', ...
    'ave_rand_sc_1','best_rand_sc_1', ...
    'ave_ar_sc_1','best_ar_sc_1', ...
    'ave_jac_sc_1','best_jac_sc_1', ...
    'ave_fm_sc_1','best_fm_sc_1');
