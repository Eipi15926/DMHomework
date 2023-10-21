#coding=utf-8
import  xml.dom.minidom
import numpy as np
from PIL import Image
import json
import os

maxx = 2500
maxy = 2500
eyesight = 2
typeSet = ['terminator','text','arrow','data','process','decision']


def draw_line(p1,p2):
    ret = []
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    for i in range(x1,x2):
        ret.append((i,int((y2-y1)/(x2-x1)*(i-x1)+y1)))
    for i in range(y1,y2):
        ret.append((int((x2-x1)/(y2-y1)*(i-y1)+x1),i))
    return ret


def draw_point(x,y,image_map):
    if x>=maxx or y >=maxy:
        print("error\n")
    for i in range(-eyesight,eyesight):
        for j in range(-eyesight,eyesight):
            xi = x + i
            yj = y + j
            if xi>=0 and xi <maxx and yj>=0 and yj<maxy:
                image_map[yj][xi]=0


def trace_to_cdlist(trace):
    tracestr = trace.firstChild.data
    tracearr = tracestr.strip().split(',')
    cnt = 0
    pointlist = []
    for xy in tracearr:
        coord = xy.strip().split()
        x = int(float(coord[0]))
        y = int(float(coord[1]))
        #print("x = {}, y = {}.".format(x, y))
        cnt = cnt + 1
        pointlist.append((x,y))
    #print(cnt)
    return pointlist


def get_img(filename,image_map,img_id):
    imagename = (filepath.split('.')[0]).split('/')[1]
    img_str = {}
    img_str['id'] = img_id
    img_str['file_name'] = imagename + '.jpg'
    img_str['coco_url'] = ""
    img_str['height'] = maxy
    img_str['width'] = maxx
    img_str['date_captured'] = '2023-10-19'
    img_str['license'] = 6 # Arbritrarily chosen
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    tracelist = root.getElementsByTagName('trace')
    #print(len(tracelist))
    for trace in tracelist:
        pointlist = trace_to_cdlist(trace)
        morepoint = []
        for i in range(0,len(pointlist)-1):
            morepoint = morepoint + draw_line(pointlist[i],pointlist[i+1])
            morepoint = morepoint + draw_line(pointlist[i+1],pointlist[i])
        allpoint = pointlist + morepoint
        for pnt in allpoint:
            draw_point(pnt[0],pnt[1],image_map)
    image = Image.fromarray(image_map).convert('RGB')
    image.save(imagename + '.jpg')
    return image_map, img_str


def get_box(tRefList,tracelist):
    allpoint = []
    xlist = []
    ylist = []
    for refIdx in tRefList:
        for trace in tracelist:
            if trace.getAttribute('id') == refIdx:
                allpoint = allpoint + trace_to_cdlist(trace)
    for point in allpoint:
        xlist.append(point[0])
        ylist.append(point[1])
    x = min(xlist)
    y = min(ylist)
    width = max(xlist) - x
    height = max(ylist) - y
    return x,y,width,height


def get_category(id,namestr):
    ctgry = {}
    ctgry['id'] = id
    ctgry['name'] = namestr
    ctgry['supercategory'] = None
    return ctgry


def inkml_to_json(anndict, ctgrydict):
    str = ""
    str = str + 'annotation' + json.dumps(anndict) + ' category' + json.dumps(ctgrydict)
    print(str)
    return str


def get_flowchart(filename, image_map, img_id):
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    tracelist = root.getElementsByTagName('trace')
    tGlist = root.getElementsByTagName('traceGroup')
    curid = 0
    annstr = []
    ctgrystr = []
    for tg in tGlist:
        annotation = {}
        categories = {}
        traceRefList = []
        elementType = ""
        for tgc in tg.childNodes:
            if tgc.nodeName == 'annotation':
                elementType = tgc.firstChild.data
            if tgc.nodeName == 'traceView':
                traceRefList.append(tgc.getAttribute('traceDataRef'))
        if elementType in typeSet:
            annotation['id'] = curid
            annotation['image_id'] = img_id
            annotation['category_id'] = curid
            x,y,w,h = get_box(traceRefList,tracelist)
            if elementType == 'arrow':
                draw_rectangle(image_map,x,y,w,h)
            annotation['segmentation'] = [x,y,x+w,y,x+w,y+h,x,y+h]
            annotation['area'] = w*h
            annotation['bbox'] = [x,y,w,h]
            annotation['iscrowd'] = 0
            categories = get_category(curid,elementType)
            # ann_json_str = ann_json_str + inkml_to_json(annotation,categories)
            annstr.append(annotation)
            ctgrystr.append(categories)
            curid = curid + 1
    #image = Image.fromarray(image_map)
    #image.show()
    return  annstr, ctgrystr


def draw_rectangle(image_map,x,y,w,h):
    pointlst = [(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)]
    for i in range(0,len(pointlst)-1):
        pointlst = pointlst + draw_line(pointlst[i],pointlst[i+1])
        pointlst = pointlst + draw_line(pointlst[i+1],pointlst[i])
    for pnt in pointlst:
        draw_point(pnt[0],pnt[1],image_map)
    return


def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


image_list = []
ann_list = []
ctgry_list = []




rootpath = 'test'
pathlist = os.listdir(rootpath)
cnt = 0
for fp in pathlist:
    print(fp)
    filepath = rootpath + '/' + fp
    image_mapp = []
    for i in range(0,maxx):
        tmp = []
        for j in range(0,maxy):
            tmp.append(255)
        image_mapp.append(tmp)
    blank_map = np.array(image_mapp)
    image_map, img_str = get_img(filepath, blank_map, cnt)
    annstr, ctgrystr = get_flowchart(filepath, image_map, cnt)
    #image_list = image_list + img_str + ','
    image_list.append(img_str)
    ann_list= ann_list + annstr
    ctgry_list = ctgry_list + ctgrystr
    cnt = cnt + 1

output_dict = {}
output_dict['info'] = ""
output_dict['images'] = image_list
output_dict['annotations'] = ann_list
output_dict['license'] = ""
output_dict['categories'] = ctgry_list
save_to_file('writerjs.json', json.dumps(output_dict, indent=4, ensure_ascii=True))