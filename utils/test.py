'''
Date of modification    : 2021.01.22
Code Summary            : realsense camera python code ìµœì¢… ë²„ì „ 
Input                   option      0 : D435i (default)
	                                1 : L515
	                                2 : D445i
'''

#####################################################
##              Import                             ##
#####################################################

import numpy as np
import cv2

from new_utils_RS import Realsense

#####################################################
##              etc                                ##
#####################################################

'''
clear the interpreter console. 
Method 1,2
'''

# import os
# os.system('cls||clear')

# print ("\033c")

#####################################################
##              Stream                             ##
#####################################################

def main():
    ss = Realsense(1)
    ss.get_Intrinsics()
    # print(ss.get_Intrinsics())
    zm = []
    while True:
        cviz, dviz = ss.output_image()
        
        print(dviz.shape)
        x,y,z = ss.get_Depth_Point(640,360)
        cv2.imshow('test', cviz)
        print('x : {}, y : {}, z : {}'.format(x, y, z))
                
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()