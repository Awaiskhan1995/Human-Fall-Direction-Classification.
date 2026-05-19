% Define the source folder with videos and the destination folder for frames
videoFolder = 'E:\fall_dataset\forward-fall'; % Change this to your video folder path
outputFolder = 'E:\Falldataset_frames\Forward_fall'; % Change this to your output folder path; % Change this to your output folder path

% Get a list of all video files in the folder
videoFiles = dir(fullfile(videoFolder, '*.mp4')); % Change the extension if needed (.avi, .mov, etc.)

% Define start and end times for frame extraction
startTime = 6; % Start time in seconds
endTime = 6.8; % End time in seconds

% Define new dimensions for the resized frames
newWidth = 320; % Example width in pixels
newHeight = 240; % Example height in pixels

% Loop through each video file
for i = 1:length(videoFiles)
    % Get the video file name and full path
    videoFile = fullfile(videoFolder, videoFiles(i).name);

    % Create a VideoReader object
    videoReader = VideoReader(videoFile);

    % Create a unique folder for this video
    [~, videoName, ~] = fileparts(videoFiles(i).name);
    outputVideoFolder = fullfile(outputFolder, videoName);
    if ~exist(outputVideoFolder, 'dir')
        mkdir(outputVideoFolder);
    end

    % Read frames between the specified times
    frameNumber = 1;
    while hasFrame(videoReader)
        % Get the current time in the video
        currentTime = videoReader.CurrentTime;

        % If within the desired time range, save the frame
        if currentTime >= startTime && currentTime <= endTime
            % Read the frame
            frame = readFrame(videoReader);

            % Resize the frame to specified dimensions
            resizedFrame = imresize(frame, [newHeight, newWidth]);

            % Save the frame as an image
            frameFileName = fullfile(outputVideoFolder, sprintf('frame_%03d.png', frameNumber));
            imwrite(resizedFrame, frameFileName);

            frameNumber = frameNumber + 1;
        elseif currentTime > endTime
            % If past the desired time range, break the loop
            break;
        else
            % Otherwise, skip the frame
            readFrame(videoReader);
        end
    end
end
