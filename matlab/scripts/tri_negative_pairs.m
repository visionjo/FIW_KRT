[a,b] = unique(cellfun(@(x) x(1:5), tt.fmdpairs.child, 'uni',false));

FIDT=table(a, tt.fmdpairs.folds(b));
nfids = length(FIDT.a);
fmdpair_table = table(zeros(length(tripairs),1), ones(length(tripairs),1),tripairs(:,1),tripairs(:,2),tripairs(:,3), 'VariableNames',{'fold','label','F','M','C'});
tfids = cellfun(@(x) x(1:5), fmdpair_table.F,'uni',false);
for x = 1:nfids
    fold = FIDT.Var2(x);
    fid = FIDT.a{x};
    
    ids = strcmp(fid,tfids);
    fmdpair_table.fold(ids) = fold;
end



fmd_negpair_table=fmdpair_table;

fmd_negpair_table.label = zeros(1,length(fmd_negpair_table.label));

label=fmd_negpair_table.label;
for x = 1:length(fmd_negpair_table.label)
    label(x) = 0;
end

fmd_negpair_table.label=label;
folds = fmd_negpair_table.folds;
fids = FIDT.a
idz2 = find(ids(~strcmp(fids,fid)));
npairs = length(fmd_negpair_table.fold);
folds = fmd_negpair_table.fold;
for f = 1:1
    ids = folds == f;
    idz = find(ids);
    fidz = FIDT.a;
    nfids = length(fidz);
    for x = 1:nfids
        fid = fidz{x};
        idz2 = find(ids(~strcmp(fids,fid)));
        fidz2 = find(ids(strcmp(fids,fid)));
        nsamps = length(fidz2);
        for z = 1:nsamps
            r = randsample(idz2, 1, 0 );
            idz2(idz2==r) = [];
            while counts(r) > 2
                r = randsample(idz2, 1, 0);
                idz2(idz2==r) = [];
                if isempty(idz2)
                    disp("ERRROR")
                end
            end
            fmd_negpair_table.child(fidz2(z)) = child(r);
            counts(r) = counts(r) + 1;
        end
    end
end


pairs = readtable('/home/jrobby/Dropbox/Families_In_The_Wild/RFIW2.0/tri-subject/info/fmd-pairs.csv','Delimiter',',');
for z = 1:nsamps
    r = randsample(idz2, 1, 0 );
    idz2(idz2==r) = [];
    while counts(r) > 2
        r = randsample(idz2, 1, 0);
        idz2(idz2==r) = [];
        if isempty(idz2)
            disp("ERRROR")
        end
    end
    fmd_negpair_table.child(fidz2(z)) = child(r);
    counts(r) = counts(r) + 1;
end


for x = 1:nfids
    fid = fidz{x};
    idz2 = find(ids(~strcmp(fids,fid)));
    fidz2 = find(ids(strcmp(fids,fid)));
    nsamps = length(fidz2)
    for z = 1:nsamps
        r = randsample(idz2, 1, 0 );
        idz2(idz2==r) = [];
        while counts(r) > 1
            r = randsample(idz2, 1, 0);
            idz2(idz2==r) = [];
            if isempty(idz2)
                disp("ERRROR")
            end
        end
        fmd_negpair_table.child(fidz2(z)) = child(r);
        counts(r) = counts(r) + 1;
    end
end

fmd_negpair_table = fmdpair_table;
fids = cellfun(@(x) x(1:5), fmdpair_table.C, 'uni',false);
counts = zeros(1,length(fmd_negpair_table.label));

for f = 1
    ids = folds == f;
    idz = find(ids);
    fids = cellfun(@(x) x(1:5), fmdpair_table.C, 'uni',false);
    fidz = unique(fids(idz));
    nfids = length(fidz);
    for x = (1:nfids)
        fid = fidz{x};
        idz2 = find(ids(~strcmp(fids,fid)));
        fidz2 = find(ids(strcmp(fids,fid)));
        nsamps = length(fidz2);
        for z = 1:nsamps
            r = randsample(idz2, 1, 0 );
            idz2(idz2==r) = [];
            while counts(r) > 3
                r = randsample(idz2, 1, 0);
                idz2(idz2==r) = [];
                if isempty(idz2)
                    disp("ERRROR")
                end
            end
            fmd_negpair_table.C(fidz2(z)) = fmdpair_table.C(r);
            counts(r) = counts(r) + 1;
        end
    end
end

fmd_negpair_table = fmdpair_table;
fids = cellfun(@(x) x(1:5), fmdpair_table.C, 'uni',false);
counts = zeros(1,length(fmd_negpair_table.label));
for f = 1:3
    ids = folds == f;
    idz = find(ids);
    fidz = unique(fids(idz));
    nfids = length(fidz);
    for x = 1:nfids
        disp(x)
        fid = fidz{x};
        idz2 = find(ids(~strcmp(fids,fid)));
        fidz2 = find(ids(strcmp(fids,fid)));
        nsamps = length(fidz2);
        for z = 1:nsamps
            r = randsample(idz2, 1, 0 );
            idz2(idz2==r) = [];
            while counts(r) > 3
                r = randsample(idz2, 1, 0);
                idz2(idz2==r) = [];
                if isempty(idz2)
                    disp("ERRROR")
                end
            end
            fmd_negpair_table.C(fidz2(z)) = fmdpair_table.C(r);
            counts(r) = counts(r) + 1;
        end
    end
end

table(a, tt.fmdpairs.folds(b))

cd Dropbox/Families_In_The_Wild/katsmile/matlab/
idir = '/home/jrobby/Dropbox/Families_In_The_Wild/RFIW2.0/FIDs/';
C = vertcat(fmd_pair_table,fmd_negpair_table);
C = vertcat(fmdpair_table,fmd_negpair_table);
C = cell(fmdpair);
fmdpairs = table2cell(C);
myToolbox.i_o.cell2csv('/home/jrobby/Dropbox/Families_In_The_Wild/RFIW2.0/tri-subject/fmdpairs.csv', fmdpairs)