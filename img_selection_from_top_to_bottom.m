% Define the folder containing the subfolders
mainFolder = 'E:\Falldataset_frames\Non-fall';

% Define the output folder outside of the main directory
outputFolder = 'E:\Fall_dataset_allframes\Non_fall';

% Create the output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Define the number of images to select from each folder
numImages = 7;

% Loop through each subfolder
folders = dir(fullfile(mainFolder, '*'));
folders = folders([folders.isdir]); % Keep only directories
folders = folders(~ismember({folders.name}, {'.', '..'})); % Remove '.' and '..'

if isempty(folders)
    disp('No subfolders found.');
    return;
end

% Initialize a counter to rename images
counter = 1;

% Loop through each subfolder
for f = 1:numel(folders)
    currentFolder = fullfile(mainFolder, folders(f).name);
    files = dir(fullfile(currentFolder, '*.png')); % Select all JPEG files in the folder

    % Check if there are enough images in the folder
    if numel(files) >= numImages
        % Select the first 'numImages' files
        selectedFiles = files(1:numImages);

        % Copy the selected files to the output folder with renamed filenames
        for j = 1:numel(selectedFiles)
            [~, name, ext] = fileparts(selectedFiles(j).name);
            newName = sprintf('image_%03d%s', counter, ext); % Rename the image
            copyfile(fullfile(currentFolder, selectedFiles(j).name), fullfile(outputFolder, newName));
            counter = counter + 1; % Increment counter
        end
    else
        disp(['Skipping folder ' folders(f).name ' due to insufficient number of images.']);
    end
end
