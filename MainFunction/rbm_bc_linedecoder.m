% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches
% restart   -- set to 1 if learning starts from beginning

function [ hiddencode ] = rbm_bc_linedecoder(inputdata, numhid)
% numdims:数据元祖的维度（初始）
% numhid:隐藏层的维度
 epsilonw      = 0.00002;   %Learning rate for weights 
 epsilonvb     = 0.00002;   %Learning rate for biases of visible units
 epsilonhb     = 0.00002;   %Learning rate for biases of hidden units
 weightcost  = 0.00004;
 initialmomentum  = 0.5;
 finalmomentum    = 0.9;
 
 [numcases, numdims]=size(inputdata); %numcases:每批数据的个数
 numbatches = 1; %numbtches:数据批数1
 %主程序定义的变量，这里重新定义，否则无法运行
 maxepoch = 5; %最大迭代次数
 epoch = 1;
 
 % p(h|v)p(v)=p(v|h)p(h)  p(v)用v表示，乘以权重得到p(h|v)，二值化得到p(h),反向得到p(v|h)
 % Initializing symmetric weights and biases.
 vishid     = 0.1*randn(numdims, numhid);
 hidbiases  = zeros(1,numhid);
 visbiases  = zeros(1,numdims); 

 poshidprobs = zeros(numcases,numhid); 
 neghidprobs = zeros(numcases,numhid); 
 posprods    = zeros(numdims,numhid); 
 negprods    = zeros(numdims,numhid); 
 vishidinc  = zeros(numdims,numhid); 
 hidbiasinc = zeros(1,numhid); 
 visbiasinc = zeros(1,numdims); 
 batchposhidprobs=zeros(numcases,numhid,numbatches); 

 for epoch = epoch:maxepoch 
   fprintf(1,'epoch %d\r',epoch); 
   errsum=0; 
   for batch = 1:numbatches 
      fprintf(1,'epoch %d batch %d\r',epoch,batch); 
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      batchdata = inputdata(:,:,batch); 
      batchdata = im2double(batchdata);
      poshidprobs = 1./(1 + exp(-batchdata*vishid - repmat(hidbiases,numcases,1)));                                       
      batchposhidprobs(:,:,batch) = poshidprobs; 
      posprods    = batchdata' * poshidprobs; 
      poshidact   = sum(poshidprobs);
      posvisact = sum(batchdata); 
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      poshidstates = poshidprobs > rand(numcases,numhid); 
      
%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      negdata = poshidstates*vishid' + repmat(visbiases,numcases,1); 
      neghidprobs = (negdata*vishid) + repmat(hidbiases,numcases,1); 
      negprods  = negdata'*neghidprobs; 
      neghidact = sum(neghidprobs); 
      negvisact = sum(negdata);
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      err = sum(sum( (batchdata-negdata).^2 ));
      errsum = err + errsum;

      if epoch>5
         momentum = finalmomentum;
      else
         momentum = initialmomentum; %0.9
      end
   
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
      vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid); 
      visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact); 
      hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact); 
      vishid = vishid + vishidinc;
      visbiases = visbiases + visbiasinc;
      hidbiases = hidbiases + hidbiasinc;
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

   end
        fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
 
 end

 poshidprobs = 1./(1 + exp(-batchdata*vishid - repmat(hidbiases,numcases,1))); 
 hiddencode = double(poshidprobs);
%  save hiddencode;
end
