function runHw5(varargin)
% runHw5 is the "main" interface that lets you execute all the 
% challenges in homework 5. It lists a set of 
% functions corresponding to the problems that need to be solved.
%
% Note that this file also serves as the specifications for the functions 
% you are asked to implement. In some cases, your submissions will be autograded. 
% Thus, it is critical that you adhere to all the specified function signatures.
%
% Before your submssion, make sure you can run runHw5('all') 
% without any error.
%
% Usage:
% runHw5                       : list all the registered functions
% runHw5('function_name')      : execute a specific test
% runHw5('all')                : execute all the registered functions

% Settings to make sure images are displayed without borders.
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

fun_handles = {@video1l, @video1r, @video2l, @video2r};

% Call test harness
runTests(varargin, fun_handles);



%%
function video1l
%-------------------
% Parameters
%-------------------
data_params.data_dir = '../images/cube/Video-1';
data_params.out_dir = '../images/cube/Video-1/tracker-l';
data_params.frame_ids = [1:176];
data_params.genFname = @(x)([sprintf('img%04dl.png', x)]);

% ****** IMPORTANT ******
% In your submission, replace the call to "chooseTarget" with actual parameters
% to specify the target of interest
% tracking_params.rect = chooseTarget(data_params);
tracking_params.rect = [102 163 45 45];

% Half size of the search window
tracking_params.search_radius = 10;
% Number of bins in the color histogram
tracking_params.bin_n = 10;    

% Pass the parameters to trackingTester (partial implementation below)
trackingTester(data_params, tracking_params);

% Take all the output frames and generate a video
generateVideo(data_params);


%%
function video1r
%-------------------
% Parameters
%-------------------
data_params.data_dir = '../images/cube/Video-1';
data_params.out_dir = '../images/cube/Video-1/tracker-r';
data_params.frame_ids = [1:176];
data_params.genFname = @(x)([sprintf('img%04dr.png', x)]);

% ****** IMPORTANT ******
% In your submission, replace the call to "chooseTarget" with actual parameters
% to specify the target of interest
% tracking_params.rect = chooseTarget(data_params);
tracking_params.rect = [102 162 47 47];

% Half size of the search window
tracking_params.search_radius = 10;
% Number of bins in the color histogram
tracking_params.bin_n = 10;    

% Pass the parameters to trackingTester (partial implementation below)
trackingTester(data_params, tracking_params);

% Take all the output frames and generate a video
generateVideo(data_params);


%%
function video2l
%-------------------
% Parameters
%-------------------
data_params.data_dir = '../images/cube/Video-2';
data_params.out_dir = '../images/cube/Video-2/tracker-l';
data_params.frame_ids = [1:120];
data_params.genFname = @(x)([sprintf('img%04dl.png', x)]);

% ****** IMPORTANT ******
% In your submission, replace the call to "chooseTarget" with actual parameters
% to specify the target of interest
% tracking_params.rect = chooseTarget(data_params);
tracking_params.rect = [174 158 47 47];

% Half size of the search window
tracking_params.search_radius = 10;
% Number of bins in the color histogram
tracking_params.bin_n = 10;    

% Pass the parameters to trackingTester (partial implementation below)
trackingTester(data_params, tracking_params);

% Take all the output frames and generate a video
generateVideo(data_params);

%%
function video2r
%-------------------
% Parameters
%-------------------
data_params.data_dir = '../images/cube/Video-2';
data_params.out_dir = '../images/cube/Video-2/tracker-r';
data_params.frame_ids = [1:120];
data_params.genFname = @(x)([sprintf('img%04dr.png', x)]);

% ****** IMPORTANT ******
% In your submission, replace the call to "chooseTarget" with actual parameters
% to specify the target of interest
% tracking_params.rect = chooseTarget(data_params);
tracking_params.rect = [106 162 45 45];

% Half size of the search window
tracking_params.search_radius = 10;
% Number of bins in the color histogram
tracking_params.bin_n = 10;    

% Pass the parameters to trackingTester (partial implementation below)
trackingTester(data_params, tracking_params);

% Take all the output frames and generate a video
generateVideo(data_params);







%%
function rect = chooseTarget(data_params)
% chooseTarget displays an image and asks the user to drag a rectangle
% around a tracking target
% 
% arguments:
% data_params: a structure contains data parameters
% rect: [xmin ymin width height]

% Reading the first frame from the focal stack
img = imread(fullfile(data_params.data_dir,...
    data_params.genFname(data_params.frame_ids(1))));
img = imresize(img, 0.25);

% Pick an initial tracking location
imshow(img);
disp('===========');
disp('Drag a rectangle around the tracking target: ');
h = imrect;
rect = round(h.getPosition);

% To make things easier, let's make the height and width all odd
if mod(rect(3), 2) == 0, rect(3) = rect(3) + 1; end
if mod(rect(4), 2) == 0, rect(4) = rect(4) + 1; end
str = mat2str(rect);
disp(['[xmin ymin width height]  = ' str]);
disp('===========');

%%
function [] = generateVideo(data_params)
    if(~exist(data_params.out_dir, 'dir'))
       disp(["WARNING: ", data_params.out_dir, " does not exist"]); 
    else
        n_frames = numel(data_params.frame_ids);
        video_filename = 'output_video.avi';
        video = VideoWriter(fullfile(data_params.out_dir, video_filename)); %create the video object
        video.FrameRate = 20;
        open(video); %open the file for writing
        for i=1:n_frames %where N is the number of images
            curr_frame_id = data_params.frame_ids(i);
            I = imread(fullfile(data_params.out_dir,data_params.genFname(curr_frame_id))); %read the next image
            writeVideo(video, I); %write the image to file
        end
        close(video); %close the file
    end
