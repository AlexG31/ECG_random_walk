#encoding:utf8
import os
import sys
import codecs
import glob
import json
import scipy
from QTdata.loadQTdata import QTloader


curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)

def fixformat(result_input_folder, output_folder):
    for ind in xrange(1, 31):
        os.mkdir(os.path.join(output_folder, 'fixed_results', 'round%d' % ind))
        result_file_pattern = os.path.join(result_input_folder, 'round%d' % ind, 'results', '*.json')
        files = glob.glob(result_file_pattern)

        for result_file in files:
            file_name = os.path.split(result_file)[-1]
            record_name = file_name.split('.')[0]

            print 'Processing: ',record_name
            with open(result_file, 'r') as fin:
                data = json.load(fin)
                data = dict(result = data, record_name = record_name)


            output_file_path = os.path.join(output_folder, 'fixed_results', 'round%d' % ind, file_name)
            with open(output_file_path, 'w') as fout:
                json.dump(data, fout, indent = 4)

def crop_data_for_swt(rawsig):
    '''Padding zeros to make the length of the signal to 2^N.'''
    # crop rawsig
    base2 = 1
    N_data = len(rawsig)
    if len(rawsig)<=1:
        raise Exception('len(rawsig)={}, not enough for swt!', len(rawsig))
    crop_len = base2
    while base2<N_data:
        if base2*2>=N_data:
            crop_len = base2*2
            break
        base2*=2
    # Extending this signal input with its tail element.
    if N_data< crop_len:
        rawsig += [rawsig[-1],]*(crop_len-N_data)
    return rawsig


def post_p_processing(result_input_folder, output_folder):
    '''Fix the result formats and do post-p processing.'''
    from randomwalk.post_p.post_p import post_p_wt
    import copy
    
    qt = QTloader()
    
    for ind in xrange(1, 31):
        round_result_folder = os.path.join(output_folder, 'post_p_results', 'round%d' % ind)
        os.mkdir(round_result_folder)
        result_file_pattern = os.path.join(result_input_folder, 'round%d' % ind, 'results', '*.json')
        files = glob.glob(result_file_pattern)

        for result_file in files:
            file_name = os.path.split(result_file)[-1]
            record_name = file_name.split('.')[0]

            print 'Processing: ',record_name
            with open(result_file, 'r') as fin:
                data = json.load(fin)
                
                # Post-p processing
                sig = qt.load(record_name)
                raw_sig = sig['sig']
                # Resample to 500.0Hz
                print 'Loaded signal.'
                raw_sig = crop_data_for_swt(raw_sig)
                raw_sig = scipy.signal.resample(raw_sig, len(raw_sig) * 2)
                print 'Resampling done.'
                annots = filter(lambda x:x[1][0]!='P', data[0])
                P_annots = filter(lambda x:x[1][0]=='P', data[0])
                P_annots = copy.deepcopy(P_annots)
                
                # Resample annots
                P_annots = [[x[0] * 2, x[1]] for x in P_annots]
                P_annots = post_p_wt(raw_sig, P_annots, 500.0)
                # Resample annots
                P_annots = [[x[0] / 2.0, x[1]] for x in P_annots]

                print 'P-post processing done.'
                annots.extend(P_annots)

                # Append 1st lead result
                annot_list = [annots, ]

                # 2nd lead ECG
                raw_sig = sig['sig2']
                raw_sig = crop_data_for_swt(raw_sig)
                raw_sig = scipy.signal.resample(raw_sig, len(raw_sig) * 2)
                annots = filter(lambda x:x[1][0]!='P', data[1])
                P_annots = filter(lambda x:x[1][0]=='P', data[1])
                P_annots = copy.deepcopy(P_annots)
                # Resample annots
                P_annots = [[x[0] * 2, x[1]] for x in P_annots]
                P_annots = post_p_wt(raw_sig, P_annots, 500.0)
                # Resample annots
                P_annots = [[x[0] / 2.0, x[1]] for x in P_annots]
                annots.extend(P_annots)

                annot_list.append(annots)

                data = dict(result = annot_list, record_name = record_name)


            output_file_path = os.path.join(round_result_folder, file_name)
            with open(output_file_path, 'w') as fout:
                json.dump(data, fout, indent = 4)


if __name__ == '__main__':
    # Convert round result for evaluation
    round_result_folder = os.path.join(curfolderpath, 'hiking2lead')
    # fixformat(round_result_folder, round_result_folder)
    post_p_processing(round_result_folder, round_result_folder)
