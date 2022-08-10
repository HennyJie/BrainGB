function myerode(mask_file,new_mask_file)
% This code is written by Liang Zhan (zhan.liang@ucla.edu)
% Input variables
% 1) make_file name (nifti format, no gz format)
% 2) new_mask_file name (this is output in nifti format)


nii=load_untouch_nii(mask_file);
mask=double(nii.img); 
dim1=size(mask,1); dim2=size(mask,2); dim3=size(mask,3);
for zs=1:dim1
   mask(zs,:,:)=bwmorph(squeeze(mask(zs,:,:)),'erode');
end
for zs=1:dim2
   mask(:,zs,:)=bwmorph(squeeze(mask(:,zs,:)),'erode');
end
for zs=1:dim3
mask(:,:,zs)=bwmorph(mask(:,:,zs),'erode');
end
nii.img=mask;
save_untouch_nii(nii,new_mask_file);
end
