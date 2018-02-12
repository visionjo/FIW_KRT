%% Script to encode facial images via VGG_Face D-CNN model.
%
% fiw_setup;
%% Set params
run('matconvnet-1.0-beta20/matlab/vl_setupnn.m');
%params = FIW_Params;
feat_type = 'vgg_new';
%% Load data (faces)
source_dir = '/home/jrobby/Desktop/PIDs/cropped_faces2/';%params.devdir;
feats_bin = '/home/jrobby/Desktop/PIDs/features_new/';%params.devdir;
imset = imageSet(source_dir,'recursive');

% find index
mind_ind = length(source_dir);


nfaces = sum([imset.Count]);

% prepare input and output
ibins = cat(2,[imset(:).ImageLocation])';
% obins = strcat(params.feats_bin,feat_type,cellfun(@(x) x(mind_ind:end-4), ibins,'uni',false));
obins = strcat(feats_bin,feat_type,cellfun(@(x) x(mind_ind:end-4), ibins,'uni',false));
% obins = strrep(obins,[feat_type '/'],[feat_type '/F0601/']);

fbins = strcat(obins,'.mat');

odirs = strcat(cellfun(@fileparts,obins,'uni',false),'/');
cellfun(@mkdir,unique(odirs));



%% Prepare CNN
model_path = 'matconvnet-1.0-beta20/matlab/models/vgg-face.mat';
modref = 'vgg_face-fc7';

net = load(model_path) ;

batch_size = 10;
out_dim = 4096;
cropped_dim=224;
image_dim = 224;


averageImg = [129.1863,104.7624,93.5940] ;

tic
failed_ind = zeros(1,nfaces);
% par
for x =1:nfaces
    
    try
        fprintf(1,'Deep Feature %d / %d\n',x,nfaces);
        img = (imread(ibins{x}));
        im_ = imresize(img, net.meta.normalization.imageSize(1:2)) ;
        im_ = bsxfun(@minus,single(im_),net.meta.normalization.averageImage) ;
        
        res = vl_simplenn(net, im_) ;
        feat = squeeze(gather(res(end-3).x));
%         myToolbox.par
        save(fbins{x},'feat');
        
    catch
        failed_ind(x) = 1
        
    end
end
toc
%
% feats = cell(1,nfaces);
% fbins = strcat(obins,'-fc7-feats.mat');
% for y = 1:nfaces
%     tmp = load(fbins{y});
%     feats{y} = tmp.fc7_ft;
% end
%
% feats = cat(2,feats{:});
