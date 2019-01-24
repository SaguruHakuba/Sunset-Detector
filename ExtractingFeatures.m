function [X] = ExtractingFeatures(img)
[height,width,a]=size(img);
X =zeros(1,294);
for row=1:7
    for col=1:7
        subimg = img((floor((row-1)*height/7)+1):floor((row)*height/7),(floor((col-1)*width/7)+1):floor((col)*width/7),:);
        R = double(subimg(:,:,1));
        G = double(subimg(:,:,2));
        B = double(subimg(:,:,3));
        L = R+G+B;
        S = R-B;
        T = R-2*G+B;
%         subimg(:,:,1) = L/3;
%         subimg(:,:,2) = (S+256)/2;
%         subimg(:,:,3) = (T+512)/4;
        X((row-1)*6+(col-1)*42+1) = mean(mean(L));
        X((row-1)*6+(col-1)*42+2) = std(L(:));
        X((row-1)*6+(col-1)*42+3) = mean(mean(S));
        X((row-1)*6+(col-1)*42+4) = std(S(:));
        X((row-1)*6+(col-1)*42+5) = mean(mean(T));
        X((row-1)*6+(col-1)*42+6) = std(T(:));
%         figure(100)
%         imshow(subimg);
%         pause;
        
    end
end