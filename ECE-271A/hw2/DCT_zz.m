function vec = DCT_zz(DCT, order)
% This function flattens an image region into feature vector, then finds X
    dim = size(DCT,1) * size(DCT,2);
    flatten_vec = zeros(dim, 1);
    for r = 1:size(DCT,1)
        for c = 1:size(DCT,2)
            flatten_vec(order(r,c)+1) = DCT(r,c);
        end
    end
    vec = flatten_vec';
end