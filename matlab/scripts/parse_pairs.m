function [finished, pairs] =  parse_pairs(P1,P2,idir,obin, nsamps_per_fold)

fids = str2num(cell2mat(unique(cellfun(@(x) x(2:5),P1,'uni',false))));

Pairs.Var1 = P1;%T.m1;
Pairs.Var2 = P2;%T.m2;
nfold = 5;
nrelations = size(Pairs.Var2,1);

nsamps = nsamps_per_fold*nfold*2; % No. per fold x No. folds x 2 (pos/neg)
pairs = cell(nsamps,4);   % {1} fold  {2} pos/neg   {3} feat1   {4} feat2
cur_fold=1;
go = true;
x = 0;
ind = 1;
counter = 1;
do_negs = false;
while x <= nrelations-1 && go
    %% parse features for all relationships (1 image pair per exemplar)
    
    x = x + 1;
     s1= Pairs.Var1{x};
     s2= Pairs.Var2{x};
%     parent= MS_Pairs.Mother{x};
%     kid =  MS_Pairs.Son{x};
    tmp = dir([strcat(idir,'/',s1) '/*.jpg' ]);
    tmp1 = dir([strcat(idir,'/',s2) '/*.jpg' ]);
    
    if any(size(tmp,1)) &&  any(size(tmp1,1))
        parent_fpaths = strcat(s1,'/',{tmp.name});
        kid_fpaths = strcat(s2,'/',{tmp1.name});  
        parent_nsamples = length(parent_fpaths);
        kid_nsamples = length(kid_fpaths);
        
        for y = 1:parent_nsamples
            for z = 1:kid_nsamples
                if cur_fold >= ceil(counter/nsamps_per_fold)
                    pairs{ind, 1} = cur_fold;
                    pairs{ind, 2} = 1;
                    pairs{ind, 3} = parent_fpaths{y};
                    pairs{ind, 4} = kid_fpaths{z};
                    counter = counter + 1;
                    ind = ind + 1;
                else
                    do_negs = true;
                end
            end
        end
    end
    if do_negs
        %% if last index of fold, add negative samps from current fold
        cfold = cur_fold;   % add negsamps to fold (ie, count alreay incremented)
       
        inds = find([pairs{:,1}] == cfold);
        if length( unique( cellfun(@(x) x(1:5), pairs(inds,3),'uni',false))) == 1
            if cfold > 1
            inds = find([pairs{:,1}] == cfold-1);
            else
               inds = find([pairs{:,1}] == cfold+1); 
            end
        end
        %         inds = counter:(counter + nsamps_per_fold-1);
        for y = 1:nsamps_per_fold
            %% for the No. of pos samps per fold
            pairs{ind, 1} = cfold;
            pairs{ind, 2} = 0;
            r = randsample( inds, 2, 0 );
            p1 = pairs{r(1), 3} ; 
            p2 = pairs{r(2), 4} ; 
            
            while strcmp(p1(1:5),p2(1:5))
                % until families (i.e., fids) are different
                r = randsample( inds, 2,0);
                p1 = pairs{r(1), 3} ;
                p2 = pairs{r(2), 4} ;
            end
            
            pairs{ind, 3} = p1;
            pairs{ind, 4} = p2;
            ind = ind + 1;
        end
        do_negs = false;
        cur_fold = cur_fold + 1;
        if cur_fold == nfold + 1
            go = false;
        end
    end
    
end
finished = 1;
if isempty(pairs{end,end}), finished = 0; end
save(obin,'pairs');


% return;
% sets = [];
% for i = 1:length(imset)
%     loc = {};
%     set = imset(x);
%     iset = [];
% %     if 
%     for y = 1:set.Count
%         loc{length(loc) + 1} = set.ImageLocation{y};
%     end
%     iset.ImageLocation = loc;
%     iset.Description = set.Description;
%     isets = [isets; iset];
%     
%         
% end

% save('./pair_lists2/gm-gd.mat','pairs')
% save('~/Dropbox/kinship/Families_In_The_Wild/pairs2/s-s_pairs.mat','pairs')
% save('results/HOG/father_son/father_son.mat','pairs')
