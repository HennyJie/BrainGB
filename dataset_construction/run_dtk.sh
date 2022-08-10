#!/bin/bash


export PATH="$PATH:/ifshome/lzhan/software/dtk/"
export DSI_PATH="/ifshome/lzhan/software/dtk/matrices/"

export FSLDIR="/usr/local/fsl-4.1.4_64bit"
    . ${FSLDIR}/etc/fslconf/fsl.sh

data=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/1157_EC.nii.gz
bvec=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/1157.bvecs
bval=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/dti.bvals
FA=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/dti1/dti_FA_mask_modi.nii.gz
temp=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/temp/
output=/ifs/loni/9k/hardi/lzhan/UIC_RUTH/example/1157/dtk/
rm -R ${temp}
mkdir -p ${temp}
mkdir -p ${output}
cp ${data} ${temp}


############ diffusion toolkit ###############################################################################################

python -c "import sys; print('\n'.join(' '.join(c) for c in zip(*(l.split() for l in sys.stdin.readlines() if l.strip()))))" < ${bvec} > ${temp}/bvecs_4dt.bvec

dti_recon ${data} ${output}/dti -gm ${temp}/bvecs_4dt.bvec -b 800 -b0 auto -no_eigen -no_exp -ot nii

dti_tracker ${output}/dti ${output}/tract_tmp.trk -fact -at 60 -iy -m ${FA} 0.15 1 -it nii
spline_filter ${output}/tract_tmp.trk 1 ${output}/dti_fact_tracts.trk
rm ${output}/tract_tmp.trk

# rm ${output}/*.dat
