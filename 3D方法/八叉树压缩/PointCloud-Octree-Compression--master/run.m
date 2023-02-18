clear
% Read file
filename = 'C:\Cauli_private\点云压缩技术\八叉树压缩\PointCloud-Octree-Compression--master\Data3D\0000000001.ply';
quanfilePath = strcat(filename,'enc.ply');  %连接字符串
binPath = strcat(filename,'bin');
p = pcread(filename);   %读取点云文件
pointNum = p.Count;
points = p.Location;
before_points=points;

% Quantization
qs = 1e-4;% qs must be integer 
points = round((points - min(points))/qs);
pt = unique(points,'rows');   %找到位置唯一值，剔除重复点

% Save file after quantization
if ~exist(quanfilePath,'file')    %文件是否存在
    pcwrite(pointCloud(pt),quanfilePath);  %将pcwrite坐标格式写成pcd格式，pointcloud返回一个由坐标表示的点云对象
end
fprintf('input file: %s\n',filename);
fprintf('quantized  file Path: %s\n',quanfilePath);
fprintf('encoding points: %d\n',pointNum);
save('pathfile.mat','quanfilePath','filename','binPath');

% Generate octree
[code,Octree] = GenOctree(pt);
% Codes = dec2hex(code);
% disp(["codes:",Codes])
% dlmwrite(strcat(filename,'Octree.txt'),Codes,'delimiter','');
fprintf('bpp before entropy coding:%f bit\n',length(code)*8/pointNum);
% Entropy Coding
text = code;
binsize = entropyCoding(text,binPath);
fprintf('bpp after entropy coding:%f bit\n',binsize*8/pointNum);
fprintf('bin file: %s\n',binPath);

%% Decoding
% clear
disp('decoding...')
load('pathfile.mat')
disp(['binPath: ',binPath])
fileID = fopen(binPath);
lenthtext =  fread(fileID,1,'uint32');
feqC =  fread(fileID,255,'uint8');
bin =  fread(fileID,'ubit1');
fclose(fileID);
% Entropy decoding
feq = double(feqC(feqC~=0));
dtext = arithdeco(bin,feq,lenthtext);
feqT = find(feqC);
%查找非零元素的索引
dtext = feqT(dtext);
%条件为false时引发错误
assert(isequal(dtext,text))

% Decode Octree
ptRec = qs*DeOctree(dtext);
pcshow(ptRec)


% Evaluate
disp('evaluate...')
decodPath = strcat(filename,'dec.ply');
pcwrite(pointCloud(single(ptRec)),decodPath);
Cmd=['C:\Cauli_private\点云压缩技术\八叉树压缩\PointCloud-Octree-Compression--master\pc_error.exe' ,' -a ',quanfilePath ,' -b ',decodPath, ' -r ','1023']; %psnr = 10log10(3*p^2/max(mse)) e.g. p = 1023
system(Cmd);





