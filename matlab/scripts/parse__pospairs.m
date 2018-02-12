function pairs =  parse__pospairs(P1,P2,idir)
%% Parse positive face pairs
% Given set of pairs (P1 and P2) and data dir then return all combinations 
% of face pairs for each pair.
npairs = length(P1);
ind = 1;
pairs1 = {};
pairs2 = {};
for x = 1:npairs
    %% parse features for all relationships (1 image pair per exemplar)
    tmp = dir([strcat(idir,'/',P1{x}) '/*.jpg' ]);
    tmp1 = dir([strcat(idir,'/',P2{x}) '/*.jpg' ]);
    if ~any(size(tmp,1)) ||  ~any(size(tmp1,1)), continue;end
    
    fpaths1 = strcat(P1{x},'/',{tmp.name});
    fpaths2 = strcat(P2{x},'/',{tmp1.name}); 
    n1 = length(fpaths1);
    n2 = length(fpaths2);
    for y = 1:n1
        for z = 1:n2
            pairs1{ind} = fpaths1{y};
            pairs2{ind} = fpaths2{z};
            
            ind = ind + 1;
        end
    end
end
pairs = [pairs1' pairs2'];