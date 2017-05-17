#!/usr/bin/python
#coding:utf-8

import cv2
from sound_ana import pitch_ana
from sound_ana import open_sound 
from movie_ana import facedetector_dlib 

frame_cnt = 0

bgcol = ( 255, 255, 255 )
dcol  = (   0, 140, 255 )



def do_with_movie() :
    global frame_cnt
    cap = cv2.VideoCapture('/mnt/hgfs/common/S1050005.MP4')
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter( '/mnt/hgfs/common/S1050005_mod2.avi', fourcc, 30.0, (1920,1080))
    while( cap.isOpened() ) :
        ret, frame = cap.read()
        if ret == True :
            key, err = pitch_ana( frame_cnt )
            cv2.rectangle( frame, ( 40,  20 ), ( 480, 270 ), bgcol, -1 )
            cv2.putText( frame, key, ( 40, 200 ), cv2.FONT_HERSHEY_DUPLEX, 6, dcol, 3 )

            frame, s = facedetector_dlib( frame )
#           if frame_cnt > 200 :
#               frame, s = facedetector_dlib( frame )
            cv2.imshow( 'face landmark detector', frame )
            out.write( frame )
            frame_cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break
        else :
            break
    cap.release()
    cv2.destroyAllWindows()
    out.release()



def do_with_still() :
    frame = cv2.imread( '/mnt/hgfs/common/S1050005.MP4_000007970.jpg' )
    frame, s = facedetector_dlib( frame )
    cv2.imshow( 'face landmark detector', frame )
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':

    open_sound( 'output.wav' )

#	do_with_still()
    do_with_movie()
