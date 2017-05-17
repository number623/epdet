#!/usr/bin/python
#coding:utf-8

import numpy as np
import pitch_detect as pd
import wave

#
# constans & variables
#

KEY_BUFFER_SIZE = 4
SILENCE = "---"
p_ana_pos = 0
P_ANA_SAMPLES = 1470
BASE_FREQ = 442.0

FPS = 29.970

key_def = SILENCE
key_buffer = [ SILENCE ] * KEY_BUFFER_SIZE

frame_cnt = 0

#
# get Key name from frequency
#
def getKey ( Hz ) :
    cent = 1200 * np.log2( Hz / BASE_FREQ ) + 5700
    cent_r100 = round( cent, -2 )
#   print " max( amp ) : ", max( amp ), " amp.index : ", amp.index( max( amp ) ), " hz : ", Hz, " cent : ", cent, " cent_r100 : ", cent_r100
    octave = ( int ) ( cent_r100 / 1200 )
    key_id = ( int ) ( cent_r100 % 1200 ) / 100
    key_array = [ 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B', 'H']
    key = key_array[ key_id ] + str( octave )
    err = ( int ) ( cent - cent_r100 )
    return key, err


#
# pitch analysis
#
def pitch_ana( frame_cnt ) :
    global key_buffer
    global key_def

    start = int( fs / FPS * frame_cnt ) * 2 
    if  start >= len( x ) :
        return SILENCE

    sz = P_ANA_SAMPLES if( ( start + ( 2 * P_ANA_SAMPLES ) ) <= len( x ) ) else ( len(x) - start ) / 2

    chunk = [0.0] * sz

    for i in range( sz ) :
        Lbuf = x[start+i*2]
        Rbuf = x[start+i*2+1]
        chunk[i] = ( float(Lbuf) + float(Rbuf) ) / 2.0
#       print Lbuf, Rbuf, chunk[i]

    Hz = pd.get_pitch( chunk, fs )
    if Hz > 0 :
        key, err = getKey( Hz )
    else :
        key = SILENCE
        err = 0
    del key_buffer[0]
    key_buffer.append( key )
    key_def = key if key_buffer.count( key ) == len( key_buffer ) else key_def
    print ( "(frame_cnt, start, len(x), key, err) = ", frame_cnt, start, len(x), key_def, err )

#   print( len(chunk), " i=",start/2, " chunk=", chunk[0], "max=", max(chunk), " min=", min(chunk), " pitch=", Hz )
#   for i in range (N) :
#       print "*", x[start+i*2], x[start+i*2+i], chunk[i], (x[start+i*2]+x[start+i*2+1])/2.0
    return key_def, err



#
# open sound file & get data on buffer
#
def open_sound ( name ) :
    global fs
    global x
    wf = wave.open( name, "r" )
    fs = wf.getframerate()
    x  = wf.readframes( wf.getnframes() )
    x  = np.frombuffer( x, dtype= "int16" )
    wf.close()
