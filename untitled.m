% Define the file path to your video
videoFilePath = '00050_H_A_BY_C1.mp4'; % Update with your video file path

% Define the time range you want to extract frames from (in seconds)
startTime = 4.5;
endTime = 5.5;

% Create a VideoReader object to read the video file
videoReader = VideoReader(videoFilePath);

% Create an output folder to save the frames
outputFolder = 'output_frames'; % Specify the output folder name
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Read frames from the specified time range and save them as image files
frameIndex = 1;
while hasFrame(videoReader)
    currentTime = videoReader.CurrentTime;
    if currentTime >= startTime && currentTime <= endTime
        frame = readFrame(videoReader);
        % Save the frame as an image file in the output folder
        frameFileName = sprintf('frame_%04d.png', frameIndex);
        imwrite(frame, fullfile(outputFolder, frameFileName));
        frameIndex = frameIndex + 1;
    elseif currentTime > endTime
        break; % Stop reading frames once we've reached the end time
    else
        readFrame(videoReader); % Skip frames before the start time
    end
end

% Display message after completion
disp('Frames extracted and saved successfully.');

% Close the VideoReader object
delete(videoReader);
