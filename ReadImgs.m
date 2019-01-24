folders = [
    "images/images/train/sunset/", 
    "images/images/train/nonsunset/", 
    "images/images/test/sunset/", 
    "images/images/test/nonsunset/",
    "images/images/validate/sunset/",
    "images/images/validate/nonsunset/"
];
X = [];
Y=[];
for i = 1:max(size(folders))
    files = dir(fullfile(char(folders(i)),'*.jpg'));
    for j = 1:max(size(files))
        img = imread(strcat(char(folders(i)),files(j).name));
        newX = ExtractingFeatures(img);
        X = [X;newX];
        Y = [Y; 2*contains(folders(i),"/sunset")-1];
    end
end

features = normalizeFeatures01(X)
save('features.mat', 'X', 'Y','features');
    

