
% 指定根目錄
root_dir = 'F:\laboratory\Graduation_thesis\data\MNI_CN';
output_dir = 'F:\laboratory\Graduation_thesis\data\output_similar'; % 輸出資料夾
if ~exist(output_dir, 'dir')
    mkdir(output_dir); % 如果輸出資料夾不存在，則建立
end

% 找出所有 sub-NR1_xxx 受試者資料夾
subject_folders = dir(fullfile(root_dir, 'sub-NR1_*'));
fprintf('總共有 %d 個受試者資料夾。\n', length(subject_folders));

% 遍歷每個受試者資料夾
for i = 1:length(subject_folders)
    subject_folder = fullfile(root_dir, subject_folders(i).name);
    
    % 確保當前項目是資料夾
    if ~subject_folders(i).isdir
        continue;
    end
    
    % 生成檔案路徑
    fct_tensor_file = fullfile(subject_folder, 'T1_to_MNI_FCT_registration.nii.gz');
    tensor_file = fullfile(subject_folder, 'T1_to_MNI_tensor_registration.nii.gz');
    fct_fa_file = fullfile(subject_folder,'T1_to_MNI_FCT_FA_registration.nii.gz');
    fa_file = fullfile(subject_folder, 'T1_to_MNI_FA_registration.nii.gz');
    mask_file = fullfile(subject_folder, 'T1_to_MNI_T1_finalmask_registration.nii.gz');

    
    % 確保所有檔案存在
    %if ~isfile(fct_tensor_file) || ~isfile(tensor_file) || ...
    %   ~isfile(fct_fa_file) || ~isfile(fa_file) || ~isfile(mask_file)
    %    fprintf('檔案缺失於 %s，跳過此受試者。\n', subject_folders(i).name);
    %    continue;
    %end

    % 讀取影像檔案
    T1 = load_untouch_nii(fct_tensor_file);
    T2 = load_untouch_nii(tensor_file);
    T1_FA = load_untouch_nii(fct_fa_file);
    T2_FA = load_untouch_nii(fa_file);
    brain_mask = load_untouch_nii(mask_file);
    
    % 提取影像資料
    Img_mask = brain_mask.img;
    ind = find(Img_mask ~= 0 & ~isnan(Img_mask));
    Mask_xyz = zeros(length(ind), 3);
    [Mask_xyz(:,1), Mask_xyz(:,2), Mask_xyz(:,3)] = ind2sub(size(Img_mask), ind);
    nvox = size(Mask_xyz, 1);
    
    % 重新塑形影像資料
    nvol = size(T1.img, 4); % 6 orientations
    tmp_d1 = reshape(T1.img, [], nvol);
    tmp_d2 = reshape(T2.img, [], nvol);
    d1 = tmp_d1(ind, :);
    d2 = tmp_d2(ind, :);
    d1_FA = T1_FA.img(ind);
    d2_FA = T2_FA.img(ind);
    
    % 計算相似度
    simi_d = NaN(nvox, 1);
    for n = 1:nvox
        if sum(d1(n, :) == 0) ~= nvol && sum(d2(n, :) == 0) ~= nvol
            Mat_d1 = [d1(n,1), d1(n,4), d1(n,5); d1(n,4), d1(n,2), d1(n,6); d1(n,5), d1(n,6), d1(n,3)];
            Mat_d2 = [d2(n,1), d2(n,2), d2(n,3); d2(n,2), d2(n,4), d2(n,5); d2(n,3), d2(n,5), d2(n,6)];
            Mat_d1 = Mat_d1 / sum(sum(Mat_d1));
            Mat_d2 = Mat_d2 / sum(sum(Mat_d2));
            dif = Mat_d1 - Mat_d2;
            euc_dis = sqrt(trace(dif^2));
            simi_d(n, 1) = (1 / euc_dis) * d1_FA(n) * d2_FA(n);
        end
    end
    
    % 處理 Inf 值
    simi_d(isinf(simi_d)) = NaN;
    
    % 重新映射到三維影像空間
    B2 = NaN(size(Img_mask));
    for n = 1:nvox
        B2(Mask_xyz(n,1), Mask_xyz(n,2), Mask_xyz(n,3)) = simi_d(n);
    end
    
    % 儲存結果影像
    brain_mask.img = B2;
    brain_mask.hdr.dime.datatype = 16; % float
    brain_mask.hdr.dime.bitpix = 32;
    output_file = fullfile(output_dir, [subject_folders(i).name, '_DTIFCT_weighted_similar.nii.gz']);
    save_untouch_nii(brain_mask, output_file);
    
    fprintf('完成受試者 %s 的處理，結果儲存於 %s。\n', subject_folders(i).name, output_file);
end

fprintf('所有受試者處理完成！\n');

   