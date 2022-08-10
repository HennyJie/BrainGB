function compute_matrix_Dec_11_2015(trk, ROI,outputname,threshold)
% This code is written by Liang Zhan (zhan.liang@gmail.com)
% There are four input variables
% First one is trk---whole brain tractography (dtk format)
% Second one is ROI--- label (1,2,3,.....)
% third is outputname, which is output brain network name (matlab mat format)
% last one is threshold, which is to reomve false positive fibers (those shorter filbers)
trk='dti_fact_tracts.trk';
ROI='label_in_DTI_dil0.nii';
outputname='test.mat'
threshold=10

if ~exist(outputname)
mytrks = read_trk_for_camino_mod( trk );
whos
if isstr(threshold)
   threshold=str2double(threshold);
end
fibers=read_fibers_matlab(mytrks,threshold);


% fibers(ptnum<threshold,:,:)=[];
% ptnum(ptnum<threshold)=[];
% [max(ptnum),min(ptnum)]
nii=load_untouch_nii(ROI);
rootnii=nii;
img=nii.img;
for i=1:max(img(:))
% for i=70:71
    myimg=img;   
    myimg(myimg~=i)=0;
    rootnii.img=myimg;
    nii_vector{i}=rootnii;
end

% matrix=zeros(max(img(:)), max(img(:)));
matrix=fiber_roi_filter_AND(fibers,2, nii_vector);
save(outputname,'matrix');
% matrix=matrix+matrix';
end
end

function tracks = read_trk_for_camino_mod( filename )

% tracks = read_trk( filename)
% read_trk reads fiber tracks output from trackvis
% Input:
%       filename: Name of track file output from trackvis
%                 The extension of file is .trk
% Output:
%       tracks is a matlab structure.
%       It contains following fields:
%           header: is a matlab structure. It contains all header information 
%                   requires to visualize fiber tracks in trackvis.
%           fiber: is a matlab cell. Each cell corresponds to a single fiber.
%                  Each fiber is a matlab structure. It contains following fields;
%                  num_points: Number of points in a fiber tracks.
%                  points    : is a num_points X 3 matrix, each row
%                              contains (x, y, z) coordinate location in
%                              mm coordinate space.
%
% For details about header fields and fileformat see:
% http://www.trackvis.org/docs/?subsect=fileformat
%
%
% Example;
% 
% tracks = read_trk('hardiO10.trk');
% tracks = 
% 
%     header: [1x1 struct]
%      fiber: {[1x1 struct]}
%
% written by Sudhir K Pathak
% Date: March 10 2009
% for PghBC2009 competition 2009 url:http://sfcweb.lrdc.pitt.edu/pbc/2009/

% NOTE: This program reads a binary fiber tracking file output from TrackVIS in native format
% If you reading .trk file on big endian machine change fopen function:
% fid = fopen(filename ,'r', 'ieee-le'); 

%
% $Id: read_trk.m,v 1.2 2009/11/11 17:07:06 skpathak Exp $
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modified by Yan Jin 2011/02/04 22:15
% The output has been modified to the voxel coordinate system starting from
% zero
% Orientation: X: Right->Left, Y: Anterior->Posterior, Z: Inferior->Superior 

% modified by Yan Jin 2012/02/09
% TrackVis uses LPS, needs to convert to LAS first, then apply the vox-to-ras to obtain the physical space. 
% all the data processed after 11/28/2013 should use this program instead
% of read_trk_for_camino.m


try
    
    fid = fopen(filename ,'r');
    
    tracks.header.id_string                  = fread(fid,6,'char=>char');
    tracks.header.dim                        = fread(fid,3,'int16');
    tracks.header.voxel_size                 = fread(fid,3,'float');
    tracks.header.origin                     = fread(fid,3,'float');
    tracks.header.n_scalars                  = fread(fid,1,'int16');
    tracks.header.scalar_name                = fread(fid,[10,20],'char=>char');
    tracks.header.n_properties               = fread(fid,1,'int16');
    tracks.header.property_name              = fread(fid,[10,20],'char=>char');
    tracks.header.vox_to_ras                 = fread(fid,[4,4],'float');
    tracks.header.reserved                   = fread(fid,444,'char=>char');
    tracks.header.voxel_order                = fread(fid,4,'char=>char');
    tracks.header.pad2                       = fread(fid,4,'char=>char');
    tracks.header.image_orientation_patient  = fread(fid,6,'float');
    tracks.header.pad1                       = fread(fid,2,'char=>char');
    tracks.header.invert_x                   = fread(fid,1,'uchar');
    tracks.header.invert_y                   = fread(fid,1,'uchar');
    tracks.header.invert_z                   = fread(fid,1,'uchar');
    tracks.header.swap_xy                    = fread(fid,1,'uchar');
    tracks.header.swap_yz                    = fread(fid,1,'uchar');
    tracks.header.swap_zx                    = fread(fid,1,'uchar');
    tracks.header.n_count                    = fread(fid,1,'int');
    tracks.header.version                    = fread(fid,1,'int');
    tracks.header.hdr_size                   = fread(fid,1,'int');

    fprintf(1,'Reading Fiber Data ...\n');
    pct = 10;
    no_fibers = tracks.header.n_count;
    for i=1:no_fibers
        tracks.fiber{i}.num_points = fread(fid,1,'int');
        dummy = zeros(tracks.fiber{i}.num_points, 3);
        for j=1:tracks.fiber{i}.num_points
            p = fread(fid,3+tracks.header.n_scalars,'float');
            % trackvis defines the voxel origin at the left bottom corner
            % LPS -> LAS
            if strcmp(tracks.header.voxel_order(1:3)','LAS')
                dummy(j,:) = p(1:3)./tracks.header.voxel_size-[0.5;0.5;0.5];
                dummy(j,2) = dummy(j,2);
            elseif strcmp(tracks.header.voxel_order(1:3)','LPS')
                dummy(j,:) = p(1:3)./tracks.header.voxel_size-[0.5;-0.5;0.5];
                dummy(j,2) = tracks.header.dim(2) - dummy(j,2);
            else
                error('The orientation of the trk file is wrong!');
            end
            % voxel space -> physical space
%             dummy(j,:) = tracks.header.vox_to_ras(1:3,1:3)*dummy(j,:)'+tracks.header.vox_to_ras(4,1:3)';
        end;
        tracks.fiber{i}.points = dummy;
    
   % progress report
    if i/no_fibers > pct/100
        fprintf(1,'\n%3d percent fibers processed...', pct);
        pct = pct + 10;
    end;
    
    if i/no_fibers==1
       fprintf(1,'\n%3d percent fibers processed...', pct);
    end

    end;
    fprintf(1,'\n');
    
    fclose(fid);
    
catch
    fprintf('Unable to access file %s\n', filename);
    tracks = [];
end
end

function fibers=read_fibers_matlab(trks,threshold)

maxpt=0;
fibnum=length(trks.fiber);

for i=1:fibnum
    if trks.fiber{i}.num_points>maxpt
       maxpt=trks.fiber{i}.num_points;
    end
end

fibers=zeros(fibnum,2,3); 
% ptnum=zeros(fibnum,1);

index=1;
for i=1:fibnum
    if trks.fiber{i}.num_points>threshold
    fibers(index,1,:)=trks.fiber{i}.points(1,:);
    fibers(index,2,:)=trks.fiber{i}.points(trks.fiber{i}.num_points,:);    
%     ptnum(index)=trks.fiber{i}.num_points;  
    index=index+1;
    end
end
fibers(index:end,:,:)=[];
end

function matrix=fiber_roi_filter_AND(fibers, ptnum, nii_vector)

% fibers_ind

fbnum=size(fibers,1);
maxptnum=size(fibers,2);

nii=nii_vector{1};
dimx=nii.hdr.dime.dim(2);
dimy=nii.hdr.dime.dim(3);
dimz=nii.hdr.dime.dim(4);
        
if  nii.filetype ~=0
    % the image is nifti format
    vox_to_phys=[nii.hdr.hist.srow_x(1:3); nii.hdr.hist.srow_y(1:3); nii.hdr.hist.srow_z(1:3)];
    offset=[nii.hdr.hist.srow_x(4); nii.hdr.hist.srow_y(4); nii.hdr.hist.srow_z(4)];
else
    % the image is analyze format, assuming LAS
    vox_to_phys=[-nii.hdr.dime.pixdim(2) 0 0; 0 nii.hdr.dime.pixdim(3) 0; 0 0 nii.hdr.dime.pixdim(4)];
    offset=[0; 0; 0];
end
        
mask=single(ones(fbnum, maxptnum, 3));
% for i=1:fbnum
%     mask(i,ptnum(i)+1:end,:)=0;
% end
        
% fibers_x=single(reshape(squeeze(fibers(:,:,1)), 1, fbnum*maxptnum)-offset(1));
% fibers_y=single(reshape(squeeze(fibers(:,:,2)), 1, fbnum*maxptnum)-offset(2));
% fibers_z=single(reshape(squeeze(fibers(:,:,3)), 1, fbnum*maxptnum)-offset(3));
fibers_x=single(reshape(squeeze(fibers(:,:,1)), 1, fbnum*maxptnum));
fibers_y=single(reshape(squeeze(fibers(:,:,2)), 1, fbnum*maxptnum));
fibers_z=single(reshape(squeeze(fibers(:,:,3)), 1, fbnum*maxptnum));
fibers_phys=[fibers_x; fibers_y; fibers_z];
% vox_ind=single(vox_to_phys\fibers_phys);
vox_ind=single(fibers_phys);
fibers_vox=single(fibers);
        
clear fibers fibers_x fibers_y fibers_z fibers_phys
        
fibers_vox(:,:,1)=single(reshape(vox_ind(1,:), fbnum, maxptnum));
fibers_vox(:,:,2)=single(reshape(vox_ind(2,:), fbnum, maxptnum));
fibers_vox(:,:,3)=single(reshape(vox_ind(3,:), fbnum, maxptnum));
fibers_vox=single((round(fibers_vox)+1).*mask);
% fibers_vox=single((ceil(fibers_vox)+1).*mask); % bad
% fibers_vox=single((floor(fibers_vox)+1).*mask);
% fibers_vox=single((fix(fibers_vox)+1).*mask);

clear vox_ind mask
        
fibers_vox_x=single(reshape(squeeze(fibers_vox(:,:,1)), fbnum*maxptnum, 1));
fibers_vox_y=single(reshape(squeeze(fibers_vox(:,:,2)), fbnum*maxptnum, 1));
fibers_vox_z=single(reshape(squeeze(fibers_vox(:,:,3)), fbnum*maxptnum, 1));
fibers_vox_vec=single([fibers_vox_x fibers_vox_y fibers_vox_z]);

clear fibers_vox_x fibers_vox_y fibers_vox_z fibers_vox;
        

for k=1:length(nii_vector)
nii=nii_vector{k}; 
% roi1_vox=[];
% for i=1:dimz
%    [ind_x, ind_y]=find(nii.img(:,:,i)>0);
%    roi1_vox=single([roi1_vox; ind_x ind_y i*ones(length(ind_x),1)]);
% end
[siz1,siz2,siz3]=ind2sub(size(nii.img),find(nii.img>0));
roi_vox{k}=[siz1,siz2,siz3];
% roi_vox{k}=roi1_vox;
end
% roi2_vox=[];
% for i=1:dimz
%     [ind_x, ind_y]=find(nii2.img(:,:,i)>0);
%     roi2_vox=single([roi2_vox; ind_x ind_y i*ones(length(ind_x),1)]);
% end

matrix=zeros(length(nii_vector),length(nii_vector));
for i=1:length(nii_vector)
% for i=7:7
    roi1_vox=roi_vox{i};
    mask1=single(ismember(fibers_vox_vec, roi1_vox, 'rows'));
    mask1=single(sum(reshape(mask1, fbnum, maxptnum),2));
	matrix(i,i)=length(find(mask1>0));
    for j=i+1:length(nii_vector)
%     for j=16:16    
        roi2_vox=roi_vox{j};
        mask2=single(ismember(fibers_vox_vec, roi2_vox, 'rows'));
        mask2=single(sum(reshape(mask2, fbnum, maxptnum),2));
        matrix(i,j)=length(find(mask1.*mask2>0));
		matrix(j,i)=matrix(i,j);
    end
end

end

