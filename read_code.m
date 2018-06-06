function code = read_code(name, image_num, code_length)
if ~exist(name, 'file')
    fprintf( [' There is no ', name, '\n'] );
    code = [];
    return;
end
fid = fopen(name, 'rb');
tic;
code = fread(fid, code_length * image_num, 'float');
code = reshape(code, [code_length image_num]);
toc;
fclose(fid);
