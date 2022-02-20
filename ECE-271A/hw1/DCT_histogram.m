function hist = DCT_histogram(vec,N)
% This function transform X vector into normalized histogram
    hist = zeros(N*N,1); % hist is a 64-dim vector, which records the occurence of different X
    for i = 1:length(vec)
        hist(vec(i)) = hist(vec(i)) + 1; % counting
    end
    hist = hist ./ size(vec,1);
end