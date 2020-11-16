function stereo_3D_reconstruction(imgL_path, imgR_path)
    % Load the stereoParameters object.
    load('calibrationSession.mat');

    % Visualize camera extrinsics.
    showExtrinsics(stereoParams);


    imgL = imread(imgL_path);
    imgR = imread(imgR_path);

    [imgLRect, imgRRect] = rectifyStereoImages(imgL, imgR, stereoParams);

    figure;
    imshow(stereoAnaglyph(imgLRect, imgRRect));
    title('Rectified Video Frames');
