#!/bin/bash
#$ -S /bin/bash
#$ written by Liang Zhan (zhan.liang@gmail.com)
#$ -o /ifs/loni/9k/hardi/lzhan/log -j y

export LD_LIBRARY_PATH=/lib64:/lib:/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
export FREESURFER_HOME="/usr/local/freesurfer-5.3.0_64bit"
export SUBJECTS_DIR=$subject_dir
source $FREESURFER_HOME/SetUpFreeSurfer.sh

dilation_factor=3;

$FREESURFER_HOME/bin/mri_vol2vol --mov brainmask.mgz --targ rawavg.mgz --regheader --o brain-in-rawavg.mgz
$FREESURFER_HOME/bin/mri_label2vol --seg aparc+aseg.mgz --temp rawavg.mgz --o aseg-in-rawavg.mgz --regheader aparc+aseg.mgz

$FREESURFER_HOME/bin/mri_convert -i brain-in-rawavg.mgz -o brain-in-rawavg.nii
$FREESURFER_HOME/bin/mri_convert -i aseg-in-rawavg.mgz -o aseg-in-rawavg.nii

###threshold labels
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -thr 1000 -uthr 1990  -sub 1000 -thr 0 cort_hemi1
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -thr 2000 -sub 2000 -thr 0 cort_hemi2

# dilate ROIS
/usr/local/fsl/bin/fslmaths cort_hemi1 -kernel box ${dilation_factor} -dilF  cort_left_dil5
/usr/local/fsl/bin/fslmaths cort_hemi2 -kernel box ${dilation_factor} -dilF  cort_right_dil5

export FSLDIR="/usr/local/fsl-4.1.4_64bit"
    . ${FSLDIR}/etc/fslconf/fsl.sh
	
    
##roi 10 -thal
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 26 -thr 36 -uthr 36 subcort_hemi1_1
##roi 11
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 26 -thr 37 -uthr 37 subcort_hemi1_2
##roi 12
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 26 -thr 38 -uthr 38 subcort_hemi1_3
##roi 13
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 26 -thr 39 -uthr 39 subcort_hemi1_4
##roi 17
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 23 -thr 40 -uthr 40 subcort_hemi1_5
##roi 18
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 23 -thr 41 -uthr 41 subcort_hemi1_6
##roi 26
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 16 -thr 42 -uthr 42 subcort_hemi1_7
##roi 28
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -add 15 -thr 43 -uthr 43 subcort_hemi1_8

/usr/local/fsl/bin/fslmaths subcort_hemi1_1 -add subcort_hemi1_2 -add subcort_hemi1_3 -add subcort_hemi1_4 -add subcort_hemi1_5 -add subcort_hemi1_6 -add subcort_hemi1_7 -add subcort_hemi1_8 subcort_FULLhemi1

############  ADD Subcoritcal regions to R
##roi 49
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 36 -uthr 36 subcort_hemi2_1
##roi 50
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 37 -uthr 37 subcort_hemi2_2
##roi 51
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 38 -uthr 38 subcort_hemi2_3
##roi 52
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 39 -uthr 39 subcort_hemi2_4
##roi 53
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 40 -uthr 40 subcort_hemi2_5
##roi 54
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 13 -thr 41 -uthr 41 subcort_hemi2_6
##roi 58
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 16 -thr 42 -uthr 42 subcort_hemi2_7
##roi 60
/usr/local/fsl/bin/fslmaths aseg-in-rawavg.nii -sub 17 -thr 43 -uthr 43 subcort_hemi2_8

/usr/local/fsl/bin/fslmaths subcort_hemi2_1 -add subcort_hemi2_2 -add subcort_hemi2_3 -add subcort_hemi2_4 -add subcort_hemi2_5 -add subcort_hemi2_6 -add subcort_hemi2_7 -add subcort_hemi2_8 subcort_FULLhemi2



# remove overlap between left cortical and right cortical (set to zero)
${FSLDIR}/bin/fslmaths cort_left_dil5 -bin left_cort_mask
${FSLDIR}/bin/fslmaths cort_right_dil5 -bin right_cort_mask
${FSLDIR}/bin/fslmaths left_cort_mask -mul right_cort_mask -mul -1 -add 1 -mul cort_left_dil5 cort_left_final
${FSLDIR}/bin/fslmaths left_cort_mask -mul right_cort_mask -mul -1 -add 1 -mul cort_right_dil5 cort_right_final

# remove overlap between cortical and subcortical (set to subcortical)
${FSLDIR}/bin/fslmaths subcort_FULLhemi1 -bin -mul -1 -add 1 -mul cort_left_final cort_left
${FSLDIR}/bin/fslmaths subcort_FULLhemi2 -bin -mul -1 -add 1 -mul cort_left cort_left2
${FSLDIR}/bin/fslmaths subcort_FULLhemi2 -bin -mul -1 -add 1 -mul cort_right_final cort_right
${FSLDIR}/bin/fslmaths subcort_FULLhemi1 -bin -mul -1 -add 1 -mul cort_right cort_right2

# combine all ROIs left 1:43 and right 44:86
${FSLDIR}/bin/fslmaths cort_left2 -add subcort_FULLhemi1 left
${FSLDIR}/bin/fslmaths cort_right2 -add subcort_FULLhemi2 right
${FSLDIR}/bin/fslmaths right -bin -mul 43 -add right -add left FS_label

${FSLDIR}/bin/fslmaths cort_right2 -bin -mul 35 -add cort_right2 -add cort_left2 cort_label_final
gunzip cort_label_final.nii.gz
gunzip FS_label.nii.gz

# register label to DTI space
${FSLDIR}/bin/flirt -in brain-in-rawavg.nii -ref dti_FA_mask_modi.nii.gz -out brain-in-DTI.nii.gz -omat TM.mat -cost corratio -dof 12  -interp nearestneighbour
${FSLDIR}/bin/flirt -in FS_label.nii.gz -ref dti_FA_mask_modi.nii.gz -out label.nii.gz -applyxfm -init TM.mat -interp nearestneighbour

gunzip label.nii.gz
