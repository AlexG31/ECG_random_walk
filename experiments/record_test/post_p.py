#encoding:utf8
import os, sys, pdb, json
import numpy as np


def compare_annot(a1, a2):
    if a1[0] != a2[0]:
        return int(a1[0] - a2[0])
    elif a2[1] == 'P' and a1[1] == 'Poffset':
        return 1
    else:
        return -1
        
def post_p(raw_sig, annots, fs):
    '''Post processing for P wave.'''
    annots = filter(lambda x: x[1][0] == 'P' or x[1] == 'Ronset', annots)
    annots.sort(key = lambda x:x, cmp = compare_annot)
    
    for ind in xrange(1, len(annots) - 1):
        pos, label = annots[ind]
        if label == 'P':
            if annots[ind-1][1] != 'Ponset' or annots[ind + 1][1] != 'Poffset':
                print 'Warning: mis-matched P in pos %d' % pos
                continue
            ponset = int(annots[ind-1][0])
            poffset = int(annots[ind + 1][0])
            

            if np.argmax(raw_sig[ponset:poffset + 1]) + ponset != pos:
                pos = int(np.argmax(raw_sig[ponset:poffset + 1]) + ponset)

            # Poffset too close to pos
            if poffset - pos < 10 / 500.0 * fs:
                right_bound = int(min(len(raw_sig), pos + 35 / 500.0 * fs + 1))
                poffset = pos + np.argmin(raw_sig[pos:right_bound])

            # Ponset too close to pos
            if abs(ponset - pos) < 10 / 500.0 * fs:
                left_bound = int(max(0, pos - 35 / 500.0 * fs))
                ponset = left_bound + np.argmin(raw_sig[left_bound:pos])



            annots[ind][0] = pos
            annots[ind-1][0] = ponset
            annots[ind + 1][0] = poffset
    return annots

            
    
