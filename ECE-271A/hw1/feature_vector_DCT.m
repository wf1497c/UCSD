function X = feature_vector_DCT(DCT)
% This function map a 64-dim vector into a scalar, which is X
% e.g. 1053 * 64 -> 1053 * 1 (matrix form)
    data_size = size(DCT,1);
    X = zeros(data_size,1);
    M = 0; I = 0;
    for i = 1:data_size
    	[M,I] = max(DCT(i,2:end));
        X(i) = I + 1; % position within the vector
    end
    
end