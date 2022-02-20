function X = X_by_DCT(DCT, order)
% This function flattens an image region into feature vector, then finds X
    dim = size(DCT,1) * size(DCT,2);
    flatten_vec = zeros(dim, 1);
    for r = 1:size(DCT,1)
        for c = 1:size(DCT,2)
            flatten_vec(order(r,c)+1) = DCT(r,c);
        end
    end
    [M,X] = max(flatten_vec(2:end));
end