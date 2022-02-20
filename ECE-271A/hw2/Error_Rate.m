function err = Error_Rate(mask)
    ground_t = imread('cheetah_mask.bmp');
    err = 0;
    for r = 1:size(mask,1)
        for c = 1:size(mask,2)
            if ground_t(r,c) ~= mask(r,c)
                err = err + 1; 
            end
        end
    end
    err = err / size(mask,1) / size(mask,2) * 100;
    disp('Error rate(%):');
    disp(err);
end