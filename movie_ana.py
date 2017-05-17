#!/usr/bin/python
#coding:utf-8

import cv2
import dlib
import numpy as np

#
# constant and variables
#

POS_ID_NOSE  = 33
POS_ID_LIP_L = 48
POS_ID_LIP_R = 54

MP_RIM_DIAMETER = 25
bgcol = ( 255, 255, 255 )
#dcol = ( 0, 223, 31 )
dcol = ( 0, 140, 255 )

mp_det_keep = [[0 for i in range(3)] for j in range(5)]


#
# mouth piece search
#
def search_mp ( mp_win ) :
    mp_win_pp = cv2.cvtColor( cv2.GaussianBlur( mp_win, (5,5), 0 ), cv2.COLOR_BGR2GRAY )
#   ret, mp_win_pp = cv2.threshold( cv2.cvtColor( mp_win, cv2.COLOR_BGR2GRAY ), 64, 255, cv2.THRESH_BINARY_INV )
    circles = cv2.HoughCircles( mp_win_pp, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=30, param2=40, minRadius=mp_win.shape[1]/6, maxRadius=mp_win.shape[1]/2)
    r_max = -1
    x_det = -1
    y_det = -1
    result = False
    if circles is not None and len( circles ) > 0 :
        circles = circles[0]
        for ( x, y, r ) in circles :
            x, y, r = int(x), int(y), int(r)
            if y > mp_win_pp.shape[0] * 0.625 :
                continue
            elif r > r_max :
                r_max = r
                x_det = x
                y_det = y
                result = True
    if result == True :
        return x_det, y_det, r_max*2, True
    else :
        return -1, -1, -1, False


#
# get affin transform matrix
#
def get_affin_matrix ( lip_l_point, lip_r_point ) :
    center = ( lip_l_point.x, lip_l_point.y )
    dist = np.linalg.norm( np.array([lip_l_point.x,lip_l_point.y]) - np.array([lip_r_point.x,lip_r_point.y]) )
    angle = np.rad2deg( np.arccos( ( lip_r_point.x - lip_l_point.x ) / dist ) )
    scale = 400 / dist
    rotation_matrix = cv2.getRotationMatrix2D( center, -angle, scale )
    return rotation_matrix, scale


#
# get affin transformed position
#
def get_affin_pos ( M, old_x, old_y ) :
    point = np.float32((old_x,old_y))
    x = point[0]
    y = point[1]
    new_x = M[0][0]*x + M[0][1]*y + np.float32(M[0][2])
    new_y = M[1][0]*x + M[1][1]*y + np.float32(M[1][2])
    return int(new_x), int(new_y)


#
# get average mouth piece position
#
def get_average_mp_pos ( mp_ary ) :
    mp_center_x_sum = 0
    mp_center_y_sum = 0
    mp_diameter_sum = 0
    valid_cnt = 0
    for i in range(len(mp_ary)) :
        if mp_ary[i][0] != 0 and mp_ary[i][1] != 0 and mp_ary[i][2] != 0 :	
            mp_center_x_sum += mp_ary[i][0]
            mp_center_y_sum += mp_ary[i][1]
            mp_diameter_sum += mp_ary[i][2]
            valid_cnt += 1
    return int( mp_center_x_sum / valid_cnt ), int( mp_center_y_sum / valid_cnt ), int( mp_diameter_sum / valid_cnt )


#
# face detection and landmark detection
#
def facedetector_dlib ( img ):
#	try:
        predictor_path = "./shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor( predictor_path )
        img_rgb = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
        dets, scores, idx = detector.run( img_rgb, 1 )
        color = ( 255, 255, 0 )
        s = ''
        if len( dets ) > 0:
            for i, rect in enumerate( dets ):
                shape = predictor( img_rgb, rect )
                nose_point = shape.part( POS_ID_NOSE ) ;
                cv2.circle( img, ( nose_point.x, nose_point.y ), 3, color, -1 ) 
                lip_l_point = shape.part( POS_ID_LIP_L ) ;
                cv2.circle( img, ( lip_l_point.x, lip_l_point.y ), 3, color, -1 ) 
                lip_r_point = shape.part( POS_ID_LIP_R ) ;
                cv2.circle( img, ( lip_r_point.x, lip_r_point.y ), 3, color, -1 ) 
                s += ( str( rect.left() ) + ' ' + str( rect.top() ) + ' ' + str( rect.right() ) + ' ' + str( rect.bottom() ) + ' ' )

                matrix, scale = get_affin_matrix( lip_l_point, lip_r_point )
                offset_x = lip_l_point.x - 60
                offset_y = lip_l_point.y - 530
                mon_nose_x, mon_nose_y = get_affin_pos( matrix, nose_point.x, nose_point.y )
                mon_nose_x -= offset_x
                mon_nose_y -= offset_y
                cv2.rectangle( img, ( 40, 290 ), ( 480, 640 ), bgcol, -1 )
                cv2.circle( img, (  60, 530 ), 4, dcol, -1 )
                cv2.circle( img, ( 460, 530 ), 4, dcol, -1 )
                cv2.line( img, ( 60, 530 ), ( 460, 530 ), dcol, 2 )
                cv2.circle( img, (  mon_nose_x, mon_nose_y ), 4, dcol, -1 )

                ### MP search window creation
                mp_win_y_start = nose_point.y
                mp_win_height  = ( 2 * ( max( lip_l_point.y, lip_r_point.y ) - nose_point.y ) )
                mp_win_y_end   = mp_win_y_start + mp_win_height 
                mp_win_x_start = lip_l_point.x
                mp_win_width   = lip_r_point.x - lip_l_point.x 
                mp_win_x_end   = mp_win_x_start + mp_win_width 
                print( "x = ", mp_win_x_start, " to ", mp_win_x_end, ", y = ", mp_win_y_start, " to ", mp_win_y_end )
                img_mp = img[ mp_win_y_start:mp_win_y_end, mp_win_x_start:mp_win_x_end ]
#               img_mp_gry = cv2.cvtColor( img_mp, cv2.COLOR_BGR2GRAY )
#               ret, img_mp_bin = cv2.threshold( img_mp_gry, 32, 255, cv2.THRESH_BINARY ) 
#               cv2.imwrite( 'mp_window.jpg', img_mp_rgb )
#               cv2.imwrite( 'output_bin.jpg', img_mp_bin )

                ### MP search
                mp_center_x, mp_center_y, mp_diameter, is_mp_detect = search_mp( img_mp )
                if is_mp_detect == True :
                    mp_center_x += mp_win_x_start
                    mp_center_y += mp_win_y_start
                    mp_det_keep.append( [mp_center_x, mp_center_y, mp_diameter] )
                    del mp_det_keep[0]
                    mp_center_x, mp_center_y, mp_diameter = get_average_mp_pos( mp_det_keep )
                    cv2.circle( img, ( mp_center_x, mp_center_y ), mp_diameter // 2, color, 2 ) 
                    cv2.circle( img, ( mp_center_x, mp_center_y ), 3, color, -1 ) 

                    mon_mp_x, mon_mp_y = get_affin_pos( matrix, mp_center_x , mp_center_y  )
                    mon_mp_x -= offset_x
                    mon_mp_y -= offset_y
                    cv2.circle( img, (  mon_mp_x, mon_mp_y ), 4, dcol, -1 )
                    cv2.circle( img, (  mon_mp_x, mon_mp_y ), int( mp_diameter // 2 * scale ), dcol, 2 )

                    ### Distance Calc.
                    mm_per_pixel = float( MP_RIM_DIAMETER ) / mp_diameter
                    dist_nose_to_mp  = np.linalg.norm( np.array([mp_center_x,mp_center_y]) - np.array([nose_point.x,nose_point.y] ) ) * mm_per_pixel
                    dist_lip_l_to_mp = np.linalg.norm( np.array([mp_center_x,mp_center_y]) - np.array([lip_l_point.x,lip_l_point.y] ) ) * mm_per_pixel
                    dist_lip_r_to_mp = np.linalg.norm( np.array([mp_center_x,mp_center_y]) - np.array([lip_r_point.x,lip_r_point.y] ) ) * mm_per_pixel
                    print( "mp rim diameter = ", MP_RIM_DIAMETER, " ---> ", mm_per_pixel , " mm/pixel" )
                    print( "dist(nose,lip_l,lip_r) = ", dist_nose_to_mp, dist_lip_l_to_mp, dist_lip_r_to_mp )

        return img, s
#	except:
#       print( "###Exception" )
#       return img, ""



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
#			if frame_cnt > 200 :
#				frame, s = facedetector_dlib( frame )
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

	### 音声ファイルの読み込み 
	wf = wave.open( 'output.wav', "r" )
	fs = wf.getframerate()
	x  = wf.readframes( wf.getnframes() )
	x  = np.frombuffer( x, dtype= "int16" )
	wf.close()

#	do_with_still()
	do_with_movie()


