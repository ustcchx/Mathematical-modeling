%% read image
change_size = 100;
im = imread('peppers.png');
%% draw 2 copies of the image
fig_1 = figure('Units', 'pixel', 'Position', [100,100,1500,1000], 'toolbar', 'none');
subplot(2,1,1); imshow(im); title({'Input image'});
im_2 = zeros(size(im) + [0 change_size 0]);
subplot(2,1,2); himg = imshow(im_2); title({'Resized Image', 'Use the blue button to resize the input image'});
hToolResize = uipushtool('CData', reshape(repmat([0 0 1], 100, 1), [10 10 3]), 'TooltipString', 'apply seam carving method to resize image', ...
                        'ClickedCallback', @(~, ~) set(himg,'cdata', seam_carve_image(im, size(im,1:2) + [0 change_size])));

function im = seam_carve_image(im, sz)
    %不同卷积核使用
    %costfunction = @(im) sum( imfilter(im, [.5 1 .5; 1 -6 1; .5 1 .5]).^2, 3 );
    costfunction = @(im) sum( imfilter(im, [1 0 -1; 2 0 -2; 1 0 -1]).^2, 3 );
    k = size(im,2) - sz(2); flag = 0;
    %这里使用报告中的小trick进行图像拉伸，直接转化为k>0情形
    if k < 0
        im = imresize(im, sz(2)/size(im, 2));
        im = imrotate(im,-90);
        k = size(im, 2) - sz(1);
        flag = 1; 
    end
    [len_1, len_2, ~] = size(im);
    %横向缩小图像
    if k >= 0
        for i = 1:k
            disp(i)
            G = costfunction(im);
            temp_matrix = zeros(size(im));
            temp_arr_1 = zeros(1, len_2);
            temp_arr_2 = zeros(1, len_2);
            G(:, [1,len_2]) = inf;
            %% find a seam in G
            for j = 1:len_1
                for t = 2:len_2-1
                    [temp_arr_2(t), temp_matrix(j, t)] = min(temp_arr_1(t-1:t+1)); 
                end
                temp_arr_1 = temp_arr_2 + G(j, 1:len_2);
            end
            temp_matrix = temp_matrix -2;
            [~,delete_index] = min(temp_arr_1(1:len_2));
            temp_im = uint8(zeros(size(im)-[0,1,0]));
            for s = len_1:-1:1
                temp_im(s, 1:len_2 - 1, :) = im(s, [1:delete_index-1, delete_index+1:len_2], :);
                delete_index = delete_index + temp_matrix(s, delete_index);
            end
            im = temp_im;
        %% remove seam from im
            len_2 = len_2 - 1;
        end
    end
    if flag == 1
        im = imrotate(im, 90);
    end
    im_2 = im;
end