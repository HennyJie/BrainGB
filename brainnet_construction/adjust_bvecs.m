function adjust_bvecs(ecclog,bvecsfile,newbvecsfile)
% This code is written by Liang Zhan (zhan.liang@ucla.edu)
% This is to rotate gradient table using eddy_correct output
% Input variables are:
%  1) ecclog --- this is one output from eddy_correct
%  2) bvecsfile --- this is bvec generated from dicom
%  3) newbvecsfile --- this is output name for adjusted bvecs 

fid=fopen(ecclog);
mat=[];
while ~feof(fid)
    % skip first three lines
    for i=1:3
      fgetl(fid);
    end
    % read four lines
    for i=1:4
     x=str2num(fgetl(fid))
    mat=[mat
         x];
    end
    % skip one line
    fgetl(fid);    
end
fclose(fid);

% read bvecs file
bvecs = load(bvecsfile);
if(size(bvecs,2)==3 && size(bvecs,1)>3)
    bvecs = bvecs';
end

% rotate bvecs
rotbvecs = zeros(size(bvecs));
for i = 1:size(bvecs,2)
    %M = mat((i-1)*4+1:i*4,:);
    %M = mat(((i-1)*3+1):i*4,:);
    %M = M(1:3,1:3);
    M = mat(1:3,1:3);
    % extract rotation matrix
    [u,s,v] = svd(M*M');
    R = inv(u*sqrt(s)*v')*M;
    
    rotbvecs(:,i) = R*bvecs(:,i);
end

save(newbvecsfile,'rotbvecs','-ascii');
end
