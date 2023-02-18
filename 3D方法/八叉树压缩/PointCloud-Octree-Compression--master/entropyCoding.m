% entropy coding
function binsize = entropyCoding(text,binPath)
%统计数组中元素出现的次数和频率
feq=tabulate(text);
feq(:,3)=[];
%归一化操作，采用0-255整数编码方式
feqInt = uint8((feq(:,2)./max(feq(:,2)))*255);
feqInt = feqInt+ uint8((feq(:,2)~=0)& (double(uint8((feq(:,2)./max(feq(:,2)))*255))==0));
feqC = feqInt;
%去除不存在的占用码
feq(feqC==0,:)=[];
feqC(feqC==0)=[];
counts = double(feqC);
%设置一个元素全为nan的的与text同容量的数组
temp = nan(size(text));
%为了与counts匹配进行的数据重排
for i=1:size(feq,1)
  temp(text==feq(i,1))=i;
end
text = temp;
bin = arithenco(text,counts);
lenthtext= uint32(length(text));
fileID = fopen(binPath,'w');
fwrite(fileID,lenthtext,'uint32');
%这里记录的是还没有进行去除不存在占用码操作的原始数据
fwrite(fileID,feqInt,'uint8');
fwrite(fileID,bin,'ubit1');
fclose(fileID);
%生成的是一个二进制码格式的文件
binsize = dir(binPath); 
binsize = binsize.bytes;
end
