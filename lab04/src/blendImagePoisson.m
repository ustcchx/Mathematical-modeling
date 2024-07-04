function imret = blendImagePoisson(im1, im2, roi, targetPosition)

    % input: im1 (background), im2 (foreground), roi (in im2), targetPosition (in im1)
    %% TODO: compute blended image
    imret = im1;
    [insidepoints_tar,point_ind] = integerPointsInPolygon([targetPosition(:, 2), targetPosition(:, 1)]);
    insidepoints_roi = integerPointsInPolygon([roi(:, 2), roi(:, 1)]);
    m = size(insidepoints_tar,1);
    for plane = 1:3
        row_ind = [];
        col_ind = [];
        vals = [];
        part1 = double(im1(:, :, plane));
        part2 = double(im2(:, :, plane));
        disp(size(part2))
        b = zeros(m, 1);
        for i = 1:m
            mid = part2(insidepoints_roi(i, 1), insidepoints_roi(i, 2));
            up = part2(insidepoints_roi(i, 1) - 1, insidepoints_roi(i, 2));
            down = part2(insidepoints_roi(i, 1) + 1, insidepoints_roi(i, 2));
            left = part2(insidepoints_roi(i, 1), insidepoints_roi(i, 2) - 1);
            right =  part2(insidepoints_roi(i, 1), insidepoints_roi(i, 2) + 1);
            b(i) = up + down + left + right -4*mid;
            row_ind = [row_ind, i];
            col_ind = [col_ind, i];
            vals = [vals, -4];
            up_index = point_ind(insidepoints_tar(i,1)-1,insidepoints_tar(i,2));
            down_index = point_ind(insidepoints_tar(i,1)+1,insidepoints_tar(i,2));
            left_index = point_ind(insidepoints_tar(i,1),insidepoints_tar(i,2)-1);
            right_index = point_ind(insidepoints_tar(i,1),insidepoints_tar(i,2)+1);
            if up_index == 0
                b(i) = b(i) - part1(insidepoints_tar(i, 1)-1, insidepoints_tar(i, 2));
            else
                row_ind = [row_ind, i];
                col_ind = [col_ind, up_index];
                vals = [vals, 1];
            end
            if down_index == 0
                b(i) = b(i) - part1(insidepoints_tar(i, 1)+1, insidepoints_tar(i, 2));
            else
                row_ind = [row_ind, i];
                col_ind = [col_ind, down_index];
                vals = [vals, 1];
            end
            if left_index == 0
                b(i) = b(i) - part1(insidepoints_tar(i, 1), insidepoints_tar(i, 2)-1);
            else
                row_ind = [row_ind, i];
                col_ind = [col_ind, left_index];
                vals = [vals, 1];
            end
            if right_index == 0
                b(i) = b(i) - part1(insidepoints_tar(i, 1), insidepoints_tar(i, 2)+1);
            else
                row_ind = [row_ind, i];
                col_ind = [col_ind, right_index];
                vals = [vals, 1];
            end
        end
        Matrix = sparse(row_ind, col_ind, vals);
        x = Matrix \ b;
        for j = 1:m
            imret(insidepoints_tar(j, 1), insidepoints_tar(j, 2), plane) = uint8(x(j));
        end
    end
end

function [insidePoints, point_ind] = integerPointsInPolygon(vertices)
    minX = floor(min(vertices(:,1)));
    maxX = ceil(max(vertices(:,1)));
    minY = floor(min(vertices(:,2)));
    maxY = ceil(max(vertices(:,2)));
    
    [x, y] = meshgrid(minX:maxX, minY:maxY);
    points = [x(:), y(:)];
    vals = zeros(1, length(points));
    
    inside = inpolygon(points(:,1), points(:,2), vertices(:,1), vertices(:,2));
    
    insidePoints = points(inside, :);
    row_ind_t = points(:,1);
    col_ind_t = points(:,2);
    vals(inside) = 1:length(insidePoints);
    point_ind = sparse(row_ind_t,col_ind_t,vals);
end

