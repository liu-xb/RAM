function temp = norm_code(hashcodes)
temp=sqrt(sum(hashcodes.^2,1)) + 1e-10;
temp=bsxfun(@rdivide, hashcodes, temp);