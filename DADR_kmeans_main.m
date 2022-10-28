
clc;clear;
filename=['D014.mat';'D017.mat';'D030.mat';'D040.mat';'D043.mat';
        'D047.mat';'D143.mat';'D151.mat';'D207.mat'];

timecount = 20;
maxlayer = 14; 
var_coefficient = 0.09; %Coefficient of variation
allFilecount = size(filename,1); 
% Initialization
all_dims = zeros(allFilecount,maxlayer+2,timecount);
all_accuracy_kmeans = zeros(allFilecount,maxlayer+2,timecount);
rand_kmeans = zeros(allFilecount,maxlayer+2,timecount);
ar_kmeans = zeros(allFilecount,maxlayer+2,timecount);
jac_kmeans = zeros(allFilecount,maxlayer+2,timecount);
fm_kmeans = zeros(allFilecount,maxlayer+2,timecount);
hiddata = cell(allFilecount,timecount); 

for time=1:timecount
    fprintf('No %d times \n',time); 
    for loop = 1:allFilecount
        all_dims(loop,1,time)=loop;
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
        vdata=data;
        all_dims(loop,2,time)=dims;
        
         %% Test data with k-means clustering
        if exist('lables')
            labels=lables;
        end
        if exist('label')
            labels=label;
        end
        k = max(labels)-min(labels)+1;
        if size(labels,2)==1
            try    
                fprintf('start kmeans....\n');
                [c,costfunctionvalue, datalabels,inter] = kmeans(data,k); 
                kmeans_org_score = accuracy(labels, datalabels);
                all_accuracy_kmeans(loop,1,time) = kmeans_org_score/100;
                other_metrics = valid_external(labels, datalabels);
                rand_kmeans(loop,1,time) = other_metrics(1);
                ar_kmeans(loop,1,time) = other_metrics(2);
                jac_kmeans(loop,1,time) = other_metrics(3);
                fm_kmeans(loop,1,time) = other_metrics(4);
            catch
                all_accuracy_kmeans(loop,1,time) = -1;
                rand_kmeans(loop,1,time) = -1;
                ar_kmeans(loop,1,time) = -1;
                jac_kmeans(loop,1,time) = -1;
                fm_kmeans(loop,1,time) = -1;
                datalabels = zeros(size(data)); 
            end
        end
        
        %%The hidden code data(hidcode) is obtained by RBM training with the original data
        hidcode = rbm_bc_linedecoder(data,dims);
        hiddcode = hidcode;
        
         %% Test hidcode with k-means clustering
        k = max(labels)-min(labels)+1;
        if size(labels,2)==1
            try    
                fprintf('start kmeans....\n');
                [c,costfunctionvalue, datalabels,inter] = kmeans(hidcode,k); 
                kmeans_org_score = accuracy(labels, datalabels);
                all_accuracy_kmeans(loop,2,time) = kmeans_org_score/100;
                other_metrics = valid_external(labels, datalabels);
                rand_kmeans(loop,2,time) = other_metrics(1);
                ar_kmeans(loop,2,time) = other_metrics(2);
                jac_kmeans(loop,2,time) = other_metrics(3);
                fm_kmeans(loop,2,time) = other_metrics(4);
            catch
                all_accuracy_kmeans(loop,2,time) = -1;
                rand_kmeans(loop,2,time) = -1;
                ar_kmeans(loop,2,time) = -1;
                jac_kmeans(loop,2,time) = -1;
                fm_kmeans(loop,2,time) = -1;
                datalabels = zeros(size(data)); 
            end
        end     
        
        %% Deep adaptive dimensionality reduction
        for numlayer=1:maxlayer 
            
            %% calculate the coefficient of variation of the column vectors of the hidcode matrix and perform feature selection
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
            all_dims(loop,numlayer+2,time)=opt_dims;
            
            hiddatas = rbm_bc_linedecoder(opt_hidcodes,opt_dims);
            hidcode = hiddatas;
            
           %% k-means clustering
            if exist('lables')
                labels=lables;
            end
            if exist('label')
                labels=label;
            end
            k = max(labels)-min(labels)+1;
            if size(labels,2)==1
                try    
                    fprintf('start kmeans....\n');
                    [c,costfunctionvalue, datalabels,inter] = kmeans(hiddatas,k); 
                    kmeans_org_score = accuracy(labels, datalabels);
                    all_accuracy_kmeans(loop,numlayer+2,time) = kmeans_org_score/100;
                    other_metrics = valid_external(labels, datalabels);
                    rand_kmeans(loop,numlayer+2,time) = other_metrics(1);
                    ar_kmeans(loop,numlayer+2,time) = other_metrics(2);
                    jac_kmeans(loop,numlayer+2,time) = other_metrics(3);
                    fm_kmeans(loop,numlayer+2,time) = other_metrics(4);
                catch
                    all_accuracy_kmeans(loop,numlayer+2,time) = -1;
                    rand_kmeans(loop,numlayer+2,time) = -1;
                    ar_kmeans(loop,numlayer+2,time) = -1;
                    jac_kmeans(loop,numlayer+2,time) = -1;
                    fm_kmeans(loop,numlayer+2,time) = -1;
                    datalabels = zeros(size(hiddatas)); 
                end
            end
              
        end
        
    end
    
end

%% Calculating the average evaluation index of each layer
 cache_all_accuracy_kmeans = all_accuracy_kmeans;
 ave_accuracy_kmeans = aveMetrics(cache_all_accuracy_kmeans);
 ave_rand_kmeans = aveMetrics(rand_kmeans);
 ave_ar_kmeans = aveMetrics(ar_kmeans);
 ave_jac_kmeans = aveMetrics(jac_kmeans);
 ave_fm_kmeans = aveMetrics(fm_kmeans);
 ave_dims = aveMetrics(all_dims);

%% calculate the mean value of the best evaluation index corresponding to each dataset
 best_accuracy_kmeans = aveBestMetrics(cache_all_accuracy_kmeans,timecount);
 best_rand_kmeans = aveBestMetrics(rand_kmeans,timecount);
 best_ar_kmeans = aveBestMetrics(ar_kmeans,timecount);
 best_jac_kmeans = aveBestMetrics(jac_kmeans,timecount);
 best_fm_kmeans = aveBestMetrics(fm_kmeans,timecount);
 
%% save data
save('kmeans_DADR.mat','filename','hiddata','all_dims','all_accuracy_kmeans', ...
    'ave_accuracy_kmeans','best_accuracy_kmeans', ...
    'ave_rand_kmeans','best_rand_kmeans', ...
    'ave_ar_kmeans','best_ar_kmeans', ...
    'ave_jac_kmeans','best_jac_kmeans', ...
    'ave_fm_kmeans','best_fm_kmeans','-v7.3');