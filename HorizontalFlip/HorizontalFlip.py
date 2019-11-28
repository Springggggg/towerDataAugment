import numpy as np
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
import os
import xml.etree.ElementTree as ET
from bbox_util import *

#获取标签文件中的矩形框
def get_rect(file_path):
    a=np.loadtxt(file_path).reshape(-1,4)   #reshape是为了防止只有一行，会变成向量造成后面失败
    print(type(a))
    print(a.shape)
    return a

#输入voc格式的xml文件，转化输出voc格式的txt文件
def xml2txt(indir,outdir):
        in_file = open(indir,'r',encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        f_w = open(outdir, 'w')
        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text
            onelist=[]
            xmlbox = obj.find('bndbox')
            xn = xmlbox.find('xmin').text
            xx = xmlbox.find('xmax').text
            yn = xmlbox.find('ymin').text
            yx = xmlbox.find('ymax').text
            f_w.write(xn + ' ' + yn + ' ' + xx + ' ' + yx + ' ')
            f_w.write('\n')
        f_w.close()
        in_file.close()

#输入原始图片和标签，输出镜像后的图片和标签，标签都是voc格式的txt标签，一行对应一个框
def HorizontalFlip(img, bboxes):
    '''输入原始图片和标签，输出镜像后的图片和标签，标签都是voc格式的txt标签，一行对应一个框'''
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
    box_w = abs(bboxes[:, 0] - bboxes[:, 2])
    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w
    return img, bboxes

#做个小实验！
# img = cv2.imread("JPEGImages/000003.jpg")[:,:,::-1]
# xml2txt("Annotations\\000003.xml","000003.txt")
# b=get_rect("000003.txt")
# plotted_img = draw_rect(img, b)
# plt.imshow(plotted_img)
# plt.show()
# img_, bboxes_ = HorizontalFlip(img.copy(), b.copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()
# np.savetxt("300000.txt", bboxes_,fmt='%f',delimiter=' ')

#批量将原始图片的xml标签转化为voc格式的txt标签    -------1
'''
xmlList=os.listdir("Annotations")
print(xmlList)
for i in xmlList:
    iCompletePath=os.path.join("Annotations",i)
    outTxt=i.split(".")[0]+'.txt'
    outTxt=os.path.join("Annotations_txt",outTxt)
    print(outTxt)
    print(iCompletePath)
    xml2txt(iCompletePath,outTxt)
'''
#批量将原始图片和原始voc格式txt标签-->镜像化        --------2
'''
jpgList=os.listdir("JPEGImages")
for i in jpgList:
    inImgPath=os.path.join("JPEGImages",i)
    filename=i.split(".")[0]+'.txt'
    inTxtPath=os.path.join("Annotations_txt",filename)
    outTxtPath = os.path.join("horizontal_voc_style_txt", filename)
    outImgPath = os.path.join('horizontal_img', i)
    img = cv2.imread(inImgPath)
    b = get_rect(inTxtPath)
    img_, bboxes_ = HorizontalFlip(img.copy(), b.copy())
    np.savetxt(outTxtPath, bboxes_, fmt='%f', delimiter=' ')
    cv2.imwrite(outImgPath,img_)

'''

#将voc格式txt标签(xmin,ymin,xmax,ymax)-->转化为yolo格式标签（label，x_center,y_center,w,h）
#因为只有一类标签tower，所以都在前面加上‘0’，0代表标签tower  -------3
vocTxtDir='horizontal_voc_style_txt'
yoloTxtDir='horizontal_yolo_style_txt'
jpgDir='horizontal_img'
vocList=os.listdir(vocTxtDir)
for i in vocList:
    inPath=os.path.join(vocTxtDir,i)
    outPath=os.path.join(yoloTxtDir,i)
    jpgPath = os.path.join(jpgDir, i.split('.')[0] + '.jpg')
    print(inPath)
    img=cv2.imread(jpgPath)
    print(img.shape)
    height=img.shape[0]
    width=img.shape[1]
    a = np.loadtxt(inPath,dtype=float).reshape(-1, 4)
    f_w=open(outPath,'w')
    for i in a:
        print(i)
        x_center=((i[0]+i[2])/2.0)/width
        y_center=((i[1]+i[3])/2.0)/height
        w=(i[2]-i[0])/width
        h=(i[3]-i[1])/height
        c='0'+" "+str(x_center)+" "+str(y_center)+" "+str(w)+" "+str(h)+'\n'
        f_w.write(c)
    f_w.close()

