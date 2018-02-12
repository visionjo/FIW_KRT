% function [finished, pairs] =  prepare_pairs_fiw(P1,P2,idir,obin)
%% I/O
rootdir = '/home/jrobby/Dropbox/Families_In_The_Wild/Database/journal_data/Pairs/';
idir = [rootdir 'faces_positive/'];
odir = [rootdir 'pair_lists/'];
mkdir(odir);

%%  get list of positive face pairs
fnames = dir(strcat(idir,'*.csv'));
ifiles = strcat(idir, {fnames.name})';
ofiles = strcat(odir, strrep({fnames.name},'-faces',''))';

for f = 1:length(ifiles)
    pospairs = readtable(ifiles{f},'Delimiter',',');
    npairs = length(pospairs.p1);
    
    %% instantiate table object to store all pos and neg pairs
    pairs = table(zeros(2*npairs,1), zeros(2*npairs,1),cell(2*npairs,1), cell(2*npairs,1), 'VariableNames',{'fold', 'label', 'p1', 'p2'});
    % label top half as true and assign positive pairs to these indices
    pairs.label(1:npairs) = 1;
    pairs.p1(1:npairs) = pospairs.p1;
    pairs.p2(1:npairs) = pospairs.p2;
    
    % assign faces of p1 to negative samples to later add non-kin to p2
    pairs.p1(npairs+1:end) = pospairs.p1;
    
    
    %% get FIDs to determine fold assignment
    fids = cellfun(@(x) x(1:5),pairs.p1,'uni',false);
    % fids = str2num(cell2mat(unique(cellfun(@(x) x(1:5),pairs.p1,'uni',false))));
    %% determine fold per FID
    dir1 = dir([rootdir 'fams_fold*.csv']);
    foldfiles = strcat(rootdir, {dir1.name});
    
    nfold = length(foldfiles);
    for x = 1:nfold
        fold_fids = table2cell(readtable(foldfiles{x}, 'ReadVariableNames', false));
        
        for y = 1:length(fold_fids)
            ids = strcmp(fold_fids{y}, fids);
            pairs.fold(ids) = x;
        end
        
    end
    
    %
    % mids = cellfun(@(x) strrep(x(1:find(x=='/', 1,'last')-1),'/','.'),pairs.p1,'uni',false);
    % p1_hash = cell2mat(cellfun(@(x) str2num(strrep(strrep(x,'MID',''),'F','')),mids,'uni',false));
    %
    % unique(p1_hash);
    % npep = length(unique(p1_hash));
    %
    % [a,b] = hist(p1_hash, unique(p1_hash));
    
    %% do negs
    mids_list = cellfun(@(x) strrep(x(1:find(x=='/', 1,'last')-1),'/','.'),pospairs.p1,'uni',false);
    mids_list2 = cellfun(@(x) strrep(x(1:find(x=='/', 1,'last')-1),'/','.'),pospairs.p2,'uni',false);
    negp2 = cell(npairs,1);
    samp_count = zeros(npairs, 1); % number of times sample was used
    for x = 1:nfold
        fprintf(1, '\n\n%d fold\n',x);
        inds = find( pairs.fold(1:npairs) == x);
        nsamps = length(inds);
        folds_mids = mids_list(inds);
        folds_mids2 = mids_list2(inds);
        mids = unique(folds_mids);
        nmids = length(mids);
        for y = 1:nmids
            fprintf(1, 'mid %d / %d\n',y, nmids);
            cmids = mids{y};
            c_ids = find(strcmp(folds_mids,cmids));
            n_ids = strcmp(folds_mids,cmids)==0;
            
            cmids2 = unique(folds_mids2(c_ids));
            for z = 1:length(cmids2)
                n_ids2 = strcmp(folds_mids2,cmids2(z));
                n_ids(n_ids2) = 0;
            end
            % %         if strcmp(cmids, 'F0995.MID10')
            % %             keyboard
            % %         end
            n_ids = find(n_ids);
            n_ids2=n_ids;
            %         for z = 1:length(cmids2)
            %             c_ids2 = inds(strcmp(folds_mids2,cmids2(z)));
            %             c_ids = [c_ids; c_ids2];
            %         end
            %               c_ids = unique(c_ids);
            
            
            
            n_id_samps = length(c_ids);
            maxsamps = 1;
            for z = 1:n_id_samps
                rand_ids = randsample( n_ids(n_ids > 0), 1);
                
                while samp_count(rand_ids) > maxsamps
                    n_ids(rand_ids) = 0;
                    
                    if isempty(n_ids(n_ids > 0))
                        n_ids = n_ids2;
%                         rand_ids = randsample( n_ids, 1);
                        maxsamps = maxsamps + 1;
                    end
                    rand_ids = randsample( n_ids(n_ids > 0), 1);
%                     if isempty(rand_ids)
                        %                     n_ids = inds(strcmp(folds_mids,cmids)==0);
%                         n_ids = n_ids2;
%                         rand_ids = randsample( n_ids, 1);
%                         maxsamps = maxsamps + 1;
%                     end
                end
                negp2(inds(c_ids(z))) = pospairs.p2(inds(rand_ids));
                %                 negp22{cc} =  pospairs.p2(rand_ids);
%                 cc=cc+1;
                %                         negp2(c_ids) = pospairs.p2{rand_ids};
                samp_count(inds(rand_ids)) = samp_count(inds(rand_ids)) + 1;
            end
            
            
        end
    end
    pairs.p2(npairs+1:end) = negp2;
    writetable(pairs,ofiles{f},'delimiter',',')
    pairs = table2cell(pairs);
    save(strrep(ofiles{f},'.csv','.mat'), 'pairs', '-v7.3')
end
%
% for y = 1:npairs
%     %% for the No. of pos samps per fold
%     p1 = pospairs.p1(y,1);
%
%     r = randsample( inds, 2, 0 );
%     p1 = pairs{r(1), 3} ;
%     p2 = pairs{r(2), 4} ;
%
%     while strcmp(p1(1:5),p2(1:5))
%         % until families (i.e., fids) are different
%         r = randsample( inds, 2,0);
%         p1 = pairs{r(1), 3} ;
%         p2 = pairs{r(2), 4} ;
%     end
%
%     pairs{ind, 3} = p1;
%     pairs{ind, 4} = p2;
%     ind = ind + 1;
% end
%
%
%
% % Pairs.Var1 = P1;%T.m1;
% % Pairs.Var2 = P2;%T.m2;
%
% % nrelations = size(Pairs.Var2,1);
%
% % nsamps = nsamps_per_fold*nfold*2; % No. per fold x No. folds x 2 (pos/neg)
% % pairs = cell(nsamps,4);   % {1} fold  {2} pos/neg   {3} feat1   {4} feat2
% cur_fold=1;
% go = true;
% x = 0;
% ind = 1;
% counter = 1;
% do_negs = false;
% while x <= nrelations-1 && go
%     %% parse features for all relationships (1 image pair per exemplar)
%
%     x = x + 1;
%      s1= Pairs.Var1{x};
%      s2= Pairs.Var2{x};
% %     parent= MS_Pairs.Mother{x};
% %     kid =  MS_Pairs.Son{x};
%     tmp = dir([strcat(idir,'/',s1) '/*.jpg' ]);
%     tmp1 = dir([strcat(idir,'/',s2) '/*.jpg' ]);
%
%     if any(size(tmp,1)) &&  any(size(tmp1,1))
%         parent_fpaths = strcat(s1,'/',{tmp.name});
%         kid_fpaths = strcat(s2,'/',{tmp1.name});
%         parent_nsamples = length(parent_fpaths);
%         kid_nsamples = length(kid_fpaths);
%
%         for y = 1:parent_nsamples
%             for z = 1:kid_nsamples
%                 if cur_fold >= ceil(counter/nsamps_per_fold)
%                     pairs{ind, 1} = cur_fold;
%                     pairs{ind, 2} = 1;
%                     pairs{ind, 3} = parent_fpaths{y};
%                     pairs{ind, 4} = kid_fpaths{z};
%                     counter = counter + 1;
%                     ind = ind + 1;
%                 else
%                     do_negs = true;
%                 end
%             end
%         end
%     end
%     if do_negs
%         %% if last index of fold, add negative samps from current fold
%         cfold = cur_fold;   % add negsamps to fold (ie, count alreay incremented)
%
%         inds = find([pairs{:,1}] == cfold);
%         if length( unique( cellfun(@(x) x(1:5), pairs(inds,3),'uni',false))) == 1
%             if cfold > 1
%             inds = find([pairs{:,1}] == cfold-1);
%             else
%                inds = find([pairs{:,1}] == cfold+1);
%             end
%         end
%         %         inds = counter:(counter + nsamps_per_fold-1);
%         for y = 1:nsamps_per_fold
%             %% for the No. of pos samps per fold
%             pairs{ind, 1} = cfold;
%             pairs{ind, 2} = 0;
%             r = randsample( inds, 2, 0 );
%             p1 = pairs{r(1), 3} ;
%             p2 = pairs{r(2), 4} ;
%
%             while strcmp(p1(1:5),p2(1:5))
%                 % until families (i.e., fids) are different
%                 r = randsample( inds, 2,0);
%                 p1 = pairs{r(1), 3} ;
%                 p2 = pairs{r(2), 4} ;
%             end
%
%             pairs{ind, 3} = p1;
%             pairs{ind, 4} = p2;
%             ind = ind + 1;
%         end
%         do_negs = false;
%         cur_fold = cur_fold + 1;
%         if cur_fold == nfold + 1
%             go = false;
%         end
%     end
%
% end
% finished = 1;
% if isempty(pairs{end,end}), finished = 0; end
% save(obin,'pairs');
