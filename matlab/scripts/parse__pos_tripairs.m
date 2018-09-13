function pairs =  parse__pos_tripairs(P1,P2, P3, idir)
%% Parse positive face pairs
% Given set of pairs (P1 and P2) and data dir then return all combinations 
% of face pairs for each pair.
npairs = length(P1);
ind = 1;
pairs1 = {};
pairs2 = {};
pairs3 = {};
for x = 1:npairs
    %% parse features for all relationships (1 image pair per exemplar)
    tmp = dir([strcat(idir,'/',P1{x}) '/*.jpg' ]);
    tmp1 = dir([strcat(idir,'/',P2{x}) '/*.jpg' ]);
    tmp2 = dir([strcat(idir,'/',P3{x}) '/*.jpg' ]);
    if ~any(size(tmp,1)) ||  ~any(size(tmp1,1)) ||  ~any(size(tmp2,1)) 
        continue;
    end
    
    fpaths1 = strcat(P1{x},'/',{tmp.name});
    fpaths2 = strcat(P2{x},'/',{tmp1.name}); 
    fpaths3 = strcat(P3{x},'/',{tmp2.name}); 
    n1 = length(fpaths1);
    n2 = length(fpaths2);
    n3 = length(fpaths3);
    for y = 1:n1
        for z = 1:n2
            for r = 1:n3
                pairs1{ind} = fpaths1{y};
                pairs2{ind} = fpaths2{z};
                pairs3{ind} = fpaths3{r};
                ind = ind + 1;
            end
        end
    end
end
pairs = [pairs1' pairs2' pairs3'];