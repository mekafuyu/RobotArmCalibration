import cv2
import numpy as np
import math as mt
import time as tm

# Create a blank white image
width, height = 1080, 720
image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

width_graph, height_graph = 480, 480
image_graph = np.ones((height_graph, width_graph, 3), dtype=np.uint8) * 255  # White background

cv2.line(image_graph, (0, int(height_graph * 0.25)), (width_graph, int(height_graph * 0.25)), (0, 255, 0), 1)
cv2.putText(image_graph, '0.25', (10, int(height_graph * 0.25)), cv2.FONT_HERSHEY_DUPLEX, 
                   0.3, (0,0,0), 1, cv2.LINE_AA)
cv2.line(image_graph, (0, int(height_graph * 0.5)), (width_graph, int(height_graph * 0.5)), (0, 255, 0), 1)
cv2.putText(image_graph, '0.5', (10, int(height_graph * 0.5)), cv2.FONT_HERSHEY_DUPLEX, 
                   0.3, (0,0,0), 1, cv2.LINE_AA)
cv2.line(image_graph, (0, int(height_graph * 0.75)), (width_graph, int(height_graph * 0.75)), (0, 255, 0), 1)
cv2.putText(image_graph, '0.75', (10, int(height_graph * 0.75)), cv2.FONT_HERSHEY_DUPLEX, 
                   0.3, (0,0,0), 1, cv2.LINE_AA)



center = (int(width / 2), int(height / 2))

FRAME_RATE = 1 / 60

line_1 = [(center[0], height), (0, 0)]
line_size_1 = 300

line_end_2 = [(0, 0), (0, 0)]
line_size_2 = 300

step = 1
inc = 0.001
now = tm.time()
last = now

arm_h = 1

stop = True

# linha_alvo = 
# pontos reta: (0, 420), (1080, 420)
# 0x + 1y + 420 = 0

pts_graph = [0] * width
graph_pos = 0
record = True
desired_val = 0

arm_heigth = 0
# f = open("records.txt", "a")
while True:
    now = tm.time()
    key = cv2.waitKey(1) & 0xFF
    match key:
        case 113:
            break
        case 27:
            break
        case 112:
            stop = not stop
        case 97:
            if step <= 2:
                step += 0.001
        case 100:
            if step >= 1:
                step -= 0.001
        case 119:
            if arm_heigth < 180:
                arm_heigth += 0.1
        case 115:
            if arm_heigth > -180:
                arm_heigth -= 0.1

    if stop:
        if (step < 1 or step > 2):
            inc = -inc
        step += inc
    
    step = round(step, 3)
    img_with_line = image.copy()
    img_graph = image_graph.copy()
    
    curr_fps = now - last
    
    
    if(curr_fps < FRAME_RATE):
        continue
    cv2.putText(img_with_line, f"{1 / curr_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 
                   1, (0,0,0), 1, cv2.LINE_AA)
    last = now
    
    
    

    line_1[1] = (
        int(center[0] + line_size_1 * np.cos(step * np.pi)),
        int(height + line_size_1 * np.sin(step * np.pi)))
    
    # new_step = (1 + np.cos(2*(1 + step) * np.pi)) / 2
    
    zto = round(step - 1, 3)
    # new_step = (1 + np.abs(np.cos(np.pi * zto))) / 2
    # print(step, end='\t')
    
    
    # sqrt(1 + 4(1 - 0.5)^2) - 0.5
    # sqrt(1 + 1)
    
    
    # new_step = 0.5 * mt.sqrt(1 + 4 * (zto - 0.5) ** 2) - 0.5
    val = abs(1 - 4 * (zto - 0.5) ** 2)
    test = -0.5 * mt.sqrt(val) + 1
    # new_step = 8 * (zto - 0.5) ** 4 + 0.5
    # new_step = desired_val
    new_step = np.sin(zto * np.pi) * 0.5
    
    # test_step = 1.5 - 0.5 * np.sqrt(np.abs(1 + 4 * (zto - 0.5) ** 2))
    # print(test_step, end="\t")
    
    if graph_pos >= width_graph:
        graph_pos = 0
    
    graph_pos += 1
    
    for i in range(width_graph - 1):
        color = (0, 0, 0)
        if (i + 1 == graph_pos):
            color = (255, 255, 255)
        cv2.line(img_graph,
                 [i, int(pts_graph[i] * height_graph)],
                 [i + 1, int(pts_graph[i + 1] * height_graph)],
                 color,
                 thickness=1)
    
    # print(new_step)
    
    tr_height = 420
    tr = [
        line_1[1],
        (line_1[1][0], tr_height),
        (
            line_1[1][0] - int(np.sqrt(300 ** 2 - (line_1[1][1] - 420) ** 2))
            , tr_height
        )
    ]
    
    # new_step = desired_val * mt.pi + 0.5
    
    # tilt = np.pi * 0.1
    # for point in range(len(tr)):
    #     tr[point] = (
    #         round(tr[point][0] * mt.cos(tilt) - tr[point][1] * mt.sin(tilt)),
    #         round(tr[point][0] * mt.sin(tilt) + tr[point][1] * mt.cos(tilt)))
    
    # for point in range(len(tr)):
    #     tr[point] = (
    #         (np.rint(
    #             np.matmul(
    #                 np.array([
    #                     [np.cos(tilt), -np.sin(tilt)],
    #                     [np.sin(tilt), np.cos(tilt)],
    #                 ]),
    #                 np.array(tr[point])
    #                 )
    #             )
    #         ).astype(int)
    #     )
    
    try:
        angle = mt.atan((tr[2][1] - tr[0][1]) / (tr[2][0] - tr[0][0])) / mt.pi
    except:
        angle = 0
    
    new_step = 1 + angle - (arm_heigth/180)
    pts_graph[graph_pos] = desired_val
    
    line_2 = [line_1[1], (
        int(line_1[1][0] + line_size_2 * (np.cos(new_step * np.pi))),
        int(line_1[1][1] + line_size_2 * (np.sin(new_step * np.pi))),
    )]
        
    cv2.line(img_with_line, line_1[0], line_1[1], (255, 128, 128), thickness=5)
    cv2.circle(img_with_line, line_1[0], line_size_1, (255, 0, 0))
    
    desired_val = 1 + ((tr[2][0] - line_1[1][0]) / line_size_1 / 2)
    
    cv2.line(img_with_line, line_2[0], line_2[1], (0, 0, 0), thickness=5)
    cv2.circle(img_with_line, line_2[0], line_size_2, (255, 0, 0)) 
    
    cv2.line(img_with_line, (0, line_2[1][1]), (width, line_2[1][1]), (0, 0, 255), thickness=2)
    cv2.line(img_with_line, (0, int(height - line_size_1 * 1.5)), (width, int(height - line_size_1 * 1.5)), (0, 255, 0), thickness=2)
    
    
    
    print(zto, end="\t")
    # print(tr[0], end='\t')
    # print(tr[1], end='\t')
    # print(tr[2], end='\t\t')
    # print(step - 1, end='\t')
   
    print(f"{desired_val:02f}", end='\t')
    # print(f"{test:02f}")
    print(f"{new_step:.3f}\t{graph_pos:.3f}\t{arm_heigth:.3f}")
    
    cv2.line(img_with_line, tr[0], tr[1], (0, 255, 255), thickness=2)
    cv2.line(img_with_line, tr[1], tr[2], (0, 255, 255), thickness=2)
    cv2.line(img_with_line, tr[2], tr[0], (255, 255, 0), thickness=2)
    
    arm_range = mt.sqrt(2) * (line_size_1 + line_size_2)
    cv2.line(img_with_line, [int(width / 2 - arm_range/2), height - int(arm_range/2)], [int(width / 2 + arm_range / 2), height - int(arm_range/2)], (0, 128, 255), 1)
    cv2.line(img_with_line, [int(width / 2 - arm_range/2), height - int(arm_range/2)], [int(width / 2 + arm_range / 2), height - int(arm_range/2)], (0, 128, 255), 1)
    cv2.line(img_with_line, [int(width / 2 - arm_range/2), height - int(arm_range/2)], [int(width / 2 + arm_range / 2), height - int(arm_range/2)], (0, 128, 255), 1)
    
    
    
    # if(step > 1.999):
    #     record = False
    #     break
    
    # if(record):
    #     f.write(f"{zto},{desired_val:.3f}\n")
    
    cv2.imshow('Graph', img_graph)
    
    # sqrt((x0 - x1)^2 + (y0 - y1)^2)
    
    cv2.circle(img_with_line, tr[2], 5, (0, 0, 255), 1)
    cv2.circle(img_with_line, tr[0], 5, (0, 0, 255), 1)
    
    # cateto = mt.sqrt((tr[2][0] - tr[1][0]) ** 2 + (tr[2][1] - tr[1][1]) ** 2)
    # hipotenusa = line_size_1
    # angle = 360 - round(mt.asin(cateto / hipotenusa) * 180, 3)
    
    
    
    
    cv2.putText(img_with_line, str(angle), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 
                   1, (0,0,0), 1, cv2.LINE_AA)
    
    
    
    cv2.imshow('Line', img_with_line)
    
    
# f.close()
cv2.destroyAllWindows()
