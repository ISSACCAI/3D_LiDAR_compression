function [Codes,Octree] = GenOctree(points)
tic   %启动秒表计时器
mcode = Morton(points);
Lmax = ceil((size(mcode,2)/3));%选的是莫顿码的第二维正好是表示深度的那一维，计算得到x或者y，或者z用二进制表示的位数
pointNum = size(mcode,1);
pointID = 1:pointNum;
nodeid = 0;
proot.nodeid = nodeid;
proot.childPoint={pointID};
proot.occupancyCode=[];
proot.parent=0;
Octree(1:Lmax+1) =struct('node',[],'level',0);
Octree(1).node=proot;
% Octree(1).nodeNum = 1;
for L=1:Lmax
    Octree(L+1).level = L;
    NodeTemp(1:min([pointNum,8^(L-1)])) = struct('nodeid',nan,'childPoint',{[]},'parent',0,'occupancyCode',[]);
    nodeNum = 0;
    for node = Octree(L).node
        for ptid = node.childPoint
            PId = ptid{:};
            if isempty(PId)
               continue
            end
            PId = pointID(PId);
            nodeid=nodeid+1;
            Node.nodeid = nodeid;
            Node.childPoint=cell(1,8);  %设置一个1*8的元胞
            Node.parent=node.nodeid;
            n = L-1;
            mn = mcode(PId,1+n*3:3+n*3);
            idn = bin2dec(mn)+1;
            for i = 1:8
              Node.childPoint(i)= {PId(idn==i)};
            end
            Node.occupancyCode = ismember(8:-1:1,idn);%前者是否在后者里，返回与前者同大小的10矩阵
            nodeNum = nodeNum+1;
            NodeTemp(nodeNum)=Node;
        end
    end
    Octree(L+1).node= NodeTemp(1:nodeNum);
end
Octree(1)=[];
toc
Nodes = [Octree.node]';%共轭转置
Codes = bin2dec(num2str(cell2mat({Nodes.occupancyCode}')));  %cell2mat元胞数组转成数值数组,最后是十进制表示的
end

function mcode= Morton(A)   %多维数据转为一维数据进行编码
n = ceil(log2(max(A(:))+1));   %向上取整,A(:)是矩阵转列向量，其实就是找到矩阵A里的最大值
x = dec2bin(A(:,1),n);  %十进制转二进制，A的第一列转为n位二进制码，得到x的列数是二进制的位数
y = dec2bin(A(:,2),n);
z = dec2bin(A(:,3),n);
m = cat(3,x,y,z);%重组成一个三维矩阵（pointnum,1.3）
m = permute(m,[1,3,2]);  %重排维度里的元素
mcode = reshape(m,size(x,1),[]);  %行数是size（x,1）%其实就是x中元素的个数，reshape从低维重排
% mcode = bin2dec(mcode);
end
