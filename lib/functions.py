def is_overlap_area(gt, box):
    #order: [start x, start y, end x, end y]
    if(gt[0]<=int(box[0]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[2]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[0]) and int(box[0])<=gt[2])\
    or (int(box[0])<=gt[0] and gt[2]<=int(box[2])):
        return True
    else:
        return False

def lable_selector(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    if box_a[5] > box_b[5]:
        lable = box_a[4]
        score = box_a[5]
    else:
        lable = box_b[4]
        score = box_b[5]
    return lable, score

def bigger_box(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    lable, score = lable_selector(box_a, box_b)
    bigger_box = [min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    , max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
    , lable, score]
    return bigger_box

def is_same_obj(box, r_box, th):
    #order: [start x, start y, end x, end y]
    th_y = th // 3
    th_x = (th * 2) // 3
    r_mx = (r_box[0] + r_box[2]) // 2
    sy_dist = abs(r_box[1] - box[1])
    ey_dist = abs(r_box[3] - box[3])
    l_mx = (box[0] + box[2]) // 2
    if sy_dist<th_y and ey_dist<th_y and r_box[4] == box[4]:
        if abs(l_mx - r_mx) < th_x:
            return True
        else:
            box_size = (box[2] - box[0]) * (box[3] - box[1])
            r_box_size = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
            th_size = th * th * 9
            th_th = int(th*0.2)
            if (box_size >= th_size) and (r_box_size >= th_size)\
            and (abs(box[2] - th*9)<th_th) and (abs(r_box[0] - th*7)<th_th):
                return True
            return False
    else:
        return False

def get_close_obj(boxes, r_box, th):
    #order: [start x, start y, end x, end y, lable, score]

    # make the same object map
    obj_map = []
    new_obj = 0
    for j in range(len(boxes)):
        obj_map.append(is_same_obj(boxes[j], r_box, th))

    # change the existing object
    for j in range(len(obj_map)):
        new_obj += int(obj_map[j])
        if obj_map[j]:
            boxes[j] = bigger_box(r_box, boxes[j])
            break

    # add the none existing obj
    if new_obj == 0:
        boxes.append(r_box)

    return None