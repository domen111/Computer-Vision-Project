function trackingTester(data_params, tracking_params)
    % Useful function to get ROI from the img 
    function roi = get_ROI(img, rect)
        % just a convenience function
        xmin = rect(1);
        ymin = rect(2);
        width = rect(3);
        height = rect(4);
        roi = img(ymin:ymin+height-1, xmin:xmin+width-1,:);
    end
    
    % Verify that output directory exists
    if ~exist(data_params.out_dir, 'dir')
        fprintf(1, "Creating directory %s.\n", data_params.out_dir);
        mkdir(data_params.out_dir);
    end
    trackingbox_color = [255, 255, 0];
    % Load the first frame, draw a box on top of that frame, and save it.
    first_frame = imread(fullfile(data_params.data_dir, data_params.genFname(1)));
    first_frame = imresize(first_frame, 0.25);
    annotated_first_frame = drawBox(first_frame, tracking_params.rect, trackingbox_color, 3);
    imwrite(annotated_first_frame, fullfile(data_params.out_dir, data_params.genFname(1)));
    
    % take the ROI from the first frame and keep its histogram to match
    % later
    obj_roi = get_ROI(first_frame, tracking_params.rect);
    
    
    %------------- FILL IN CODE -----------------
    % Create the intensity histogram using the obj_roi that was extracted
    % above.
    obj_hist = histcounts(rgb2gray(obj_roi), tracking_params.bin_n);
    %------------- END FILL IN CODE -----------------

	% OPTIONAL: You can do the matching in color too.
	% Use the rgb2ind function to transform the image such that it only has a
    % fixed number of colors (much less than 256^3). If you visualize the mapped_obj 
    % image it should look similar to the obj_roi image, but with much less
    % variations in the color (the palette's size is tracking_params.bin_n).
    % The output colormap will tell you which colors were chosen to be used in 
    % the mapped_obj output image.
    % NOTE: If you want to do this, you have to do it consistently for all
    % frames!
    [mapped_obj, colormap] = rgb2ind(obj_roi, tracking_params.bin_n);
    % Create a color histogram from the mapped_obj image that has the colors
    % quantized.
    % Hint: If the mapped_obj image has Q different colors, then your
    % histogram will have Q bins, one for each color.
    obj_hist = histcounts(mapped_obj, tracking_params.bin_n);
    
    % Normalize histogram such that its sum = 1
    obj_hist = double(obj_hist) / sum(obj_hist(:));

    % Tracking loop
    % -------------    
    % initial location of tracking box
    obj_col = tracking_params.rect(1);
    obj_row = tracking_params.rect(2);
    obj_width = tracking_params.rect(3);
    obj_height = tracking_params.rect(4);
    frame_ids = data_params.frame_ids;
    mid_points = zeros(length(frame_ids), 3);
    for i_frame_id = 1:length(frame_ids)
        frame_id = frame_ids(i_frame_id);
        
        % Read current frame
        fprintf('On frame %d\n', frame_id);
        frame = imread(fullfile(data_params.data_dir, data_params.genFname(frame_id)));
        frame = imresize(frame, 0.25);
        [H, W, ~] = size(frame);
        pad_size = tracking_params.search_radius;
%         pad_frame = padarray(frame, [pad_size pad_size], 1, 'both');
        %------------- FILL IN CODE -----------------
        % extract the area over which we will search for the object
        % Hint:  This step is very similar to what you did in computeFlow
        % to extract the search_area.
        search_window_row_begin = max(1, obj_row - tracking_params.search_radius);
        search_window_row_end = min(H, obj_row + obj_height-1 + tracking_params.search_radius);
        search_window_col_begin = max(1, obj_col - tracking_params.search_radius);
        search_window_col_end = min(W, obj_col + obj_width-1 + tracking_params.search_radius);
        search_window = frame(search_window_row_begin:search_window_row_end, ...
                              search_window_col_begin:search_window_col_end, :);
        %------------- END FILL IN CODE -----------------
        % Change to grayscale
%         gray_search_window = rgb2gray(search_window);
        [mapped_search_window, colormap] = ...
            rgb2ind(search_window, tracking_params.bin_n);
        % extract each object-sized sub-region from the searched area and
        % make it a column
        candidate_windows = im2col(mapped_search_window, [obj_height obj_width], 'sliding');
        num_windows = size(candidate_windows, 2);
        % compute histograms for each candidate sub-region extracted from
        % the search window
        candidate_hists = double(zeros(tracking_params.bin_n, num_windows));
        for i = 1:num_windows
            %------------- FILL IN CODE -----------------
            % Hint: You already have done this at the beginning of this
            % function.
%             [mapped_obj, colormap] = ...
%                 rgb2ind(candidate_windows(:, ,i), tracking_params.bin_n);
%             candidate_hists(:,i) = histcounts(mapped_obj, tracking_params.bin_n);
            candidate_hists(:,i) = ...
                histcounts(candidate_windows(:,i), tracking_params.bin_n);
            %------------- END FILL IN CODE -----------------

            % Normalize histogram such that its sum = 1
            candidate_hists(:,i) = candidate_hists(:,i) / sum(candidate_hists(:,i));
        end
        
        %------------- FILL IN CODE -----------------
        
        % find the best-matching region
        % Hint: You have all the candidate histograms, and you want to find
        % the one that is the most similar to the histogram you computed
        % from the first frame

        % UPDATE the obj_row and obj_col, which tell us the location of the
        % top-left pixel of the bounding box around the object we are
        % tracking.
        corr = zeros(1, num_windows);
        for i = 1:num_windows
            corr(1,i) = xcorr(obj_hist, candidate_hists(:,i), 0, 'normalized');
        end
        corr = col2im(corr, [obj_height obj_width], size(mapped_search_window), 'sliding');
        [~, index] = max(corr(:));
        [index_row, index_col] = ind2sub(size(corr), index);
%         index_row = fix((index-1) / (size(search_window,2) - obj_width + 1)) + 1;
%         index_col = rem(index - 1, (size(search_window,2) - obj_width + 1)) + 1;
        obj_row = index_row + search_window_row_begin - 1;
        obj_col = index_col + search_window_col_begin - 1;
        
        %------------- END FILL IN CODE -----------------

        % generate box annotation for the current frame
        annotated_frame = drawBox(frame, [obj_col obj_row obj_width obj_height], trackingbox_color, 3);
        % save annotated frame in the output directory
        midx = obj_col + (obj_width+1)/2;
        midy = obj_row + (obj_height+1)/2;
        mid_points(i_frame_id, :) = [frame_id, (midx-1)*4, (midy-1)*4];
%         fprintf('%d: (%d, %d)\n', frame_id, (midx-1)*4, (midy-1)*4);
        imwrite(annotated_frame, fullfile(data_params.out_dir, data_params.genFname(frame_id)));
    end
    
    writematrix(mid_points, ...
        fullfile(data_params.out_dir, 'mid_points.csv'));
end
