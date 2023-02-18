function bin2Pcd()
 clear;
 files = dir("C:\Cauli_private\点云压缩技术\八叉树压缩\PointCloud-Octree-Compression--master\Data3D"); % 获取当前文件夹中的所有文件
 len = length(files); % 文件数量
 name=[];
 for ii=1:len % 依次遍历所有的bin文件，将其转换为pcd格式
     if (strcmp(files(ii).name, '.') == 1) ... % 跳过.以及..文件夹
     || (strcmp(files(ii).name, '..') == 1)
        continue;
     end
 
     file1D=fopen("C:\Cauli_private\点云压缩技术\八叉树压缩\PointCloud-Octree-Compression--master\Data3D\"+files(ii).name,"rb"); %获取当前文件
     [a,count]=fread(file1D,'float32') % a 文件内容。矩阵a中存储了全部的点（481072行 * 1列。每个数据占一行，每4个数据为一组）；count 文件数量
     x = a(1:4:end); %获取矩阵a中所有的x点。第1行开始，步数为4，直到最后一行
     y = a(2:4:end); % 获取矩阵a中所有的y点。第2行开始，步数为4，直到最后一行
     z = a(3:4:end); % 获取矩阵a中所有的z点。第3行开始，步数为4，直到最后一行
     intensity= a(4:4:end); % 强度（反射值）。第4行开始，步数为4，直到最后一行
     data = pointCloud([x y z],'intensity',intensity); %根据xyz坐标和强度值，转换为一个点云
     pcshow(data); %显示点云
     %hh = [files(ii).name(1:end-4),'.ply'] %文件名
     pcwrite(data,[files(ii).name(1:end-4),'.ply'],'PLYFormat','ascii'); %将点云对象存储到 当前文件名.ply文件
     img=pcread([files(ii).name(1:end-4),'.ply']); % 读取ply或pcd格式的文件
     pcshow(img); %显示点云 
     pcwrite(data,[files(ii).name(1:end-4),'.pcd']); %将点云对象存储到 当前文件名.pcd文件
 end
end