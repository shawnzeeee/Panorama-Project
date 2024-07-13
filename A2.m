%1)

%First Set
s1_im1 = imread('S1-im1.jpeg');
s1_im2 = imread('S1-im2.jpg');

s1_im1 = imresize(s1_im1,750/4032);
s1_im2 = imresize(s1_im2,750/4032);

s1_im1 = imrotate(s1_im1,270);
s1_im2 = imrotate(s1_im2,270);

s1_im1Gray = rgb2gray(s1_im1);
s1_im2Gray = rgb2gray(s1_im2);

%Second Set
s2_im1 = imread('S2-im1.jpg');
s2_im2 = imread('S2-im2.jpg');

s2_im1 = imresize(s2_im1,750/4032);
s2_im2 = imresize(s2_im2,750/4032);

s2_im1 = imrotate(s2_im1,270);
s2_im2 = imrotate(s2_im2,270);

s2_im1Gray = rgb2gray(s2_im1);
s2_im2Gray = rgb2gray(s2_im2);

%Third Set

s3_im1 = imread('S3-im1.jpg');
s3_im2 = imread('S3-im2.jpg');

s3_im1 = imresize(s3_im1,750/4032);
s3_im2 = imresize(s3_im2,750/4032);

s3_im1 = imrotate(s3_im1,270);
s3_im2 = imrotate(s3_im2,270);

s3_im1Gray = rgb2gray(s3_im1);
s3_im2Gray = rgb2gray(s3_im2);

%Fourth Set

s4_im1 = imread('S4-im1.jpg');
s4_im2 = imread('S4-im2.jpg');

s4_im1 = imresize(s4_im1,750/4032);
s4_im2 = imresize(s4_im2,750/4032);

s4_im1 = imrotate(s4_im1,270);
s4_im2 = imrotate(s4_im2,270);

s4_im1Gray = rgb2gray(s4_im1);
s4_im2Gray = rgb2gray(s4_im2);

%2)

%First Set
s1_im1_fast = my_fast_detector(s1_im1Gray);
imwrite(s1_im1_fast, 'S1-fast.png');
[res, harcors] = my_harrison_cornerness(my_fast_detector(s1_im1Gray));
localmax = imdilate(harcors, ones(3));
s1_im1_fastR = res;
imwrite(((harcors== localmax).* res), 'S1-fastR.png');

s1_im2_fast = my_fast_detector(s1_im2Gray);
s1_im2_fastR = my_harrison_cornerness(my_fast_detector(s1_im2Gray));

%Second Set
s2_im1_fast = my_fast_detector(s2_im1Gray);
[res, harcors] = my_harrison_cornerness(my_fast_detector(s2_im1Gray));
imwrite(s2_im1_fast, 'S2-fast.png');
localmax = imdilate(harcors, ones(3));
s2_im1_fastR = res;
imwrite(((harcors == localmax).* res), 'S2-fastR.png');

s2_im2_fast = my_fast_detector(s2_im2Gray);
s2_im2_fastR = my_harrison_cornerness(my_fast_detector(s2_im2Gray));

%Third Set
s3_im1_fast = my_fast_detector(s3_im1Gray);
s3_im1_fastR = my_harrison_cornerness(my_fast_detector(s3_im1Gray));

s3_im2_fast = my_fast_detector(s3_im2Gray);
s3_im2_fastR = my_harrison_cornerness(my_fast_detector(s3_im2Gray));

%Fourth Set
s4_im1_fast = my_fast_detector(s4_im1Gray);
s4_im1_fastR = my_harrison_cornerness(my_fast_detector(s4_im1Gray));

s4_im2_fast = my_fast_detector(s4_im2Gray);
s4_im2_fastR = my_harrison_cornerness(my_fast_detector(s4_im2Gray));

%3)

%First Set
saveas(SURF_feature_matcher(s1_im1, s1_im2, s1_im1_fast, s1_im2_fast),'S1-fastMatch.png')
saveas(SURF_feature_matcher(s1_im1, s1_im2, s1_im1_fastR, s1_im2_fastR),'S1-fastRMatch.png')

%Second Set
saveas(SURF_feature_matcher(s2_im1, s2_im2, s2_im1_fast, s2_im2_fast),'S2-fastMatch.png')
saveas(SURF_feature_matcher(s2_im1, s2_im2, s2_im1_fastR, s2_im2_fastR),'S2-fastRMatch.png')

%Third Set
saveas(SURF_feature_matcher(s3_im1, s3_im2, s3_im1_fast, s3_im2_fast),'S3-fastMatch.png')
saveas(SURF_feature_matcher(s3_im1, s3_im2, s3_im1_fastR, s3_im2_fastR),'S3-fastRMatch.png')

%Fourth Set
saveas(SURF_feature_matcher(s4_im1, s4_im2, s4_im1_fast, s4_im2_fast),'S4-fastMatch.png')
saveas(SURF_feature_matcher(s4_im1, s4_im2, s4_im1_fastR, s4_im2_fastR),'S4-fastRMatch.png')

%4)
%First Set

imwrite(stitch_image(s1_im1Gray, s1_im2Gray, s1_im1_fast, s1_im2_fast, s1_im1, s1_im2),'S1-panorama.png');
imwrite(stitch_image(s2_im1Gray, s2_im2Gray, s2_im1_fast, s2_im2_fast, s2_im1, s2_im2),'S2-panorama.png');
imwrite(stitch_image(s3_im1Gray, s3_im2Gray, s3_im1_fast, s3_im2_fast, s3_im1, s3_im2),'S3-panorama.png');
imwrite(stitch_image(s4_im1Gray, s4_im2Gray, s4_im1_fast, s4_im2_fast, s4_im1, s4_im2),'S4-panorama.png');
function [res] = stitch_image(I1,I2, I1Fast, I2Fast, I1Colored, I2Colored)
%Using points from fastR images
points = detectHarrisFeatures(I1Fast);
[features, points] = extractFeatures(I1,points);
images = [I1,I2];
tforms(1) = affine2d;

for n=2:length(images)
    pointsPrevious = points;
    featuresPrevious = features;
    I = I2;
    
    imageSize(n,:) = size(I);
    
% Detect and extract features from both images
    %points = detectSURFFeatures(I);
    points = detectHarrisFeatures(I2Fast);
    [features, points] = extractFeatures(I2, points);

% Match the features from both images
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

% Obtain the matched points from both images
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :); 

% Estimate the geometric transformation between the matched points
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev, 'affine','Confidence', 90, 'MaxNumTrials', 1000);
    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).T = Tinv.T * tforms(i).T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits); 

    I = I1Colored;   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(1), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(1), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    
    I = I2Colored;   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(2), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(2), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    
res = panorama;
end
function [res, resHarcor] = my_harrison_cornerness(image)
sobel = [-1 0 1; -2 0 2; -1 0 1];
gaus = fspecial('gaussian',5,1);
dog = conv2(gaus, sobel);
ix = imfilter(image,dog);
iy = imfilter(image,dog');
ix2g = imfilter(ix.*ix, gaus);
iy2g = imfilter(iy.*iy, gaus);
ixy = imfilter(ix.*iy, gaus);
harcor = ix2g.*iy2g - ixy.*ixy - 0.05 * (ix2g + iy2g).^2;
corners = (harcor) >0.0001 ;
%localmax = imdilate(harcor, ones(3));
%res = ((harcor == localmax).* corners);
res = corners;
resHarcor = harcor;
end

function [res] = my_fast_detector(image)
%Threshold
t = 35;
%shift image up
imageShiftedUp = imtranslate(image,[0, -3],'FillValues',0);

%shift image right
imageShiftedRight = imtranslate(image,[3, 0],'FillValues',0);

%shift image down
imageShiftedDown = imtranslate(image,[0, 3],'FillValues',0);

%shift image left
imageShiftedLeft = imtranslate(image,[-3, 0],'FillValues',0);

            %First conduct the high speed test and if 3 of the 4 condtions are met, check other points 
corners = ((image + t < imageShiftedUp | image - t > imageShiftedUp) + ...
         (image + t < imageShiftedRight | image - t > imageShiftedRight) + ...
          (image + t < imageShiftedDown | image - t > imageShiftedDown) + ...
          (image + t < imageShiftedLeft | image - t > imageShiftedLeft) >= 3) &...
           test_other_pixels(image,22.5,t) >= 9; %Since there are 12 remaining points, only see if 9 of them are met
res = (corners);
end

function [res] = test_other_pixels(image, angle,t)
    imageShifted = imtranslate(image,[3*cosd(angle), 3*sind(angle)],'FillValues',0);
    if(mod(angle,90) == 0)
        if(angle >= 360)
            res = 0;
        else
            res = 0 + test_other_pixels(image, angle+22.5,t);
        end
    else
        res = (image + t < imageShifted | image - t > imageShifted) + test_other_pixels(image, angle+22.5,t);
    end   
end
function [res] = SURF_feature_matcher(image1Colored, image2Colored,image1, image2)
I1=im2double(image1);
I2=im2double(image2);

% Get the Key Points
Options.upright=true;
Options.tresh=0.0001;
Ipts1=OpenSurf(I1,Options);
Ipts2=OpenSurf(I2,Options);

% Put the landmark descriptors in a matrix
D1 = reshape([Ipts1.descriptor],64,[]);
D2 = reshape([Ipts2.descriptor],64,[]);

% Find the best matches
err=zeros(1,length(Ipts1));
cor1=1:length(Ipts1);
cor2=zeros(1,length(Ipts1));
for i=1:length(Ipts1),
    distance=sum((D2-repmat(D1(:,i),[1 length(Ipts2)])).^2,1);
    [err(i),cor2(i)]=min(distance);
end

% Sort matches on vector distance
[err, ind]=sort(err);
cor1=cor1(ind);
cor2=cor2(ind);

% Make vectors with the coordinates of the best matches
Pos1=[[Ipts1(cor1).y]',[Ipts1(cor1).x]'];
Pos2=[[Ipts2(cor2).y]',[Ipts2(cor2).x]'];
Pos1=Pos1(1:30,:);
Pos2=Pos2(1:30,:);

I1 = im2double(image1Colored);
I2 = im2double(image2Colored);

% Show both images
I = zeros([size(I1,1) size(I1,2)*2 size(I1,3)]);
I(:,1:size(I1,2),:)=I1; I(:,size(I1,2)+1:size(I1,2)+size(I2,2),:)=I2;
res = figure(); imshow(I); hold on;

% Show the best matches
plot([Pos1(:,2) Pos2(:,2)+size(I1,2)]',[Pos1(:,1) Pos2(:,1)]','-');
plot([Pos1(:,2) Pos2(:,2)+size(I1,2)]',[Pos1(:,1) Pos2(:,1)]','o');
end
