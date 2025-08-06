 % Summary of this function goes here 
% computing similarity between Functional Correlation Tensor and diffusion tensor

% Input
% FCT_path:            the functional correlation tensor path 
% DTI_path:             the DTI data path (must be calculated by fsl)
% FCT_FA_path:      the FA file of FCT image
 %DTI_FA_path:      the FA file of DTI image
% mask_path:         the mask image of FCT/DT image
% out_name:          the output name   
% out_path:            the output path of similarity image

% Note!!!!!: the functional correlation tensor and diffusion tensor must be saved in same format (fsl) and in same space (diffusion space or MNI space)


% The tensor coefficient saved in fsl structured as follows:
%volumes 0-5: Dxx,Dxy,Dxz,Dyy,Dyz,Dzz


% Written by Jiajia Zhao
% /2022/06/08

% the equation used to calculated the similarity between functional correlation tensor and diffusion tensor was revised by following referrence:
% Alexander, D., Gee, J., & Bajcsy, R. (1999). Similarity Measures for Matching Diffusion Tensor Images Procedings of the British Machine Vision Conference 1999.
% Alexander, D. C., Gee, J. C., & Bajcsy, R. (1999). Transformations of and similarity measures for diffusion tensor MRI?s. In Proc. Int. Workshop Biomedical Image Registration (pp. 127-136).



%--------------------------------Main Function---------------------------------------%

    
% fprintf('Processing  : Computing Similarity between FCT and DT...... \n')


T1=load_untouch_nii('/Users/bil/Desktop/fMRI_data/CN/sub-NR1_010/sub-NR1_010_fMRI_FCT_tensor.nii.gz');
T2=load_untouch_nii('/Users/bil/Desktop/DWI_data/CN/sub-NR1_010/Tensor/sub-NR1_010_tensor.nii.gz');
T1_FA=load_untouch_nii('/Users/bil/Desktop/fMRI_data/CN/sub-NR1_010/sub-NR1_010_FCT_FA.nii.gz');
T2_FA=load_untouch_nii('/Users/bil/Desktop/DWI_data/CN/sub-NR1_010/Tensor/sub-NR1_010_FA.nii.gz');
nvol=size(T1.img,4); % 6 orientions


brain_mask=load_untouch_nii('/Users/bil/Desktop/DWI_data/CN/sub-NR1_010/sub-NR1_010_T1_finalmask.nii.gz');
Img_mask=brain_mask.img;
ind=find(Img_mask~=0 & ~isnan(Img_mask));
Mask_xyz=zeros(length(ind),3);
[Mask_xyz(:,1),Mask_xyz(:,2),Mask_xyz(:,3)]=ind2sub(size(Img_mask),ind);
nvox=size(Mask_xyz,1);

tmp_d1=reshape(T1.img,[],nvol); % FCT size * 6
tmp_d2=reshape(T2.img,[],nvol);


d1=tmp_d1(ind,:); % ind size * 6
d2=tmp_d2(ind,:);
d1_FA=T1_FA.img(ind);
d2_FA=T2_FA.img(ind);

simi_d=NaN(nvox,1); % ind size * 1, and value is NaN

parfor n=1:nvox

    % if 6 oriention's values is not 0 and empty, do...
    if sum(d1(n,:)==0)~=nvol && sum(d2(n,:)==0)~=nvol && ~isempty(d1(n,:)) && ~isempty(d2(n,:))

        % transform 6 tensor coefficents to a 3x3 symmetrical matrix     
        Mat_d1=[d1(n,1),d1(n,4),d1(n,5);d1(n,4),d1(n,2),d1(n,6);d1(n,5),d1(n,6),d1(n,3)]; % correct to MRView's set (FCT)
        Mat_d2=[d2(n,1),d2(n,2),d2(n,3);d2(n,2),d2(n,4),d2(n,5);d2(n,3),d2(n,5),d2(n,6)]; 
        Mat_d1=Mat_d1/sum(sum(Mat_d1)); % divid the sum of 6 oriention's values ; Normalization
        Mat_d2=Mat_d2/sum(sum(Mat_d2));
        % calculate similarity
        dif=Mat_d1-Mat_d2;
        euc_dis=sqrt(trace(dif^2));
        simi_d(n,1)=(1/euc_dis)*d1_FA(n)*d2_FA(n); % ** I want to know this value **
    end

end
indinf=simi_d==Inf; % what is Inf????????????
simi_d(indinf)=nan;

B2=NaN(size(Img_mask));
for n=1:nvox
    B2(Mask_xyz(n,1),Mask_xyz(n,2),Mask_xyz(n,3))=simi_d(n);
end

brain_mask.img=B2;
brain_mask.hdr.dime.datatype=16; % float 
brain_mask.hdr.dime.bitpix=32;
filename=['/Users/bil/Desktop/sub-NR1_010_DTIFCT_weighted_similar.nii.gz'];
save_untouch_nii(brain_mask,filename)
   