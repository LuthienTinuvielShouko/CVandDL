filename=fullfile(pwd,'images/im.jpg');
Img=imread(filename);
if ndims(Img)==3
    I=rgb2gray(Img);
else
    I=Img;
end
Ig=imnoise(I,'poisson');
s=GetStrelList();
s