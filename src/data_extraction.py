import scipy.io as sio
import numpy as np
from get_paths import *

# Load data from a gold standard file and store it as a dictionary.

SCALP_LONG_BIP = ['Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'F7-T3', 'T3-T5', 'T5-O1',
                  'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3',
                  'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']


def get_hfo_rates(annots, duration_s):
    channels = np.unique(annots['channel'])
    types = np.unique(annots['type'])

    hfo_rates = []
    for ch in channels:
        sel_ch = annots['channel'] == ch
        sel_hfo = annots['type'] == 'HFO'
        hfo_rate = sum(np.logical_and(sel_ch, sel_hfo))/duration_s
        hfo_rates.append(hfo_rate)

    return np.array(channels), np.array(hfo_rates)


def get_ieshfo_rates(annots, duration_s):
    channels = np.unique(annots['channel'])
    types = np.unique(annots['type'])

    ies_hfo_rates = []
    for ch in channels:
        sel_ch = annots['channel'] == ch
        sel_hfo = annots['type'] == 'HFO'
        sel_ied = annots['type'] == 'IED'
        ch_hfo_dict = {}
        for key, val in annots.items():
            ch_hfo_dict[key] = val[np.logical_and(sel_ch, sel_hfo)]

        ch_ied_dict = {}
        for key, val in annots.items():
            ch_ied_dict[key] = val[np.logical_and(sel_ch, sel_ied)]

        nr_hfo = sum(np.logical_and(sel_ch, sel_hfo))
        nr_ied = sum(np.logical_and(sel_ch, sel_ied))

        ies_hfo_sel = np.full((nr_hfo, 1), False, dtype=bool)
        for hfo_idx in range(nr_hfo):
            hfo_s = ch_hfo_dict['start_s'][hfo_idx]
            hfo_e = ch_hfo_dict['end_s'][hfo_idx]
            for ied_idx in range(nr_ied):
                ied_s = ch_ied_dict['start_s'][ied_idx]
                ied_e = ch_ied_dict['end_s'][ied_idx]
                coincident_a = hfo_s >= ied_s and hfo_s <= ied_e
                coincident_b = hfo_e >= ied_s and hfo_e <= ied_e
                coincident = coincident_a or coincident_b
                if coincident:
                    ies_hfo_sel[hfo_idx] = True
                    break

        hfo_rate = np.sum(ies_hfo_sel)/duration_s
        ies_hfo_rates.append(hfo_rate)

    return np.array(channels), np.array(ies_hfo_rates)


def get_isolhfo_rates(annots, duration_s):
    channels = np.unique(annots['channel'])
    types = np.unique(annots['type'])

    isol_hfo_rates = []
    for ch in channels:
        sel_ch = annots['channel'] == ch
        sel_hfo = annots['type'] == 'HFO'
        sel_ied = annots['type'] == 'IED'
        ch_hfo_dict = {}
        for key, val in annots.items():
            ch_hfo_dict[key] = val[np.logical_and(sel_ch, sel_hfo)]

        ch_ied_dict = {}
        for key, val in annots.items():
            ch_ied_dict[key] = val[np.logical_and(sel_ch, sel_ied)]

        nr_hfo = sum(np.logical_and(sel_ch, sel_hfo))
        nr_ied = sum(np.logical_and(sel_ch, sel_ied))

        isol_hfo_sel = np.full((nr_hfo, 1), True, dtype=bool)
        for hfo_idx in range(nr_hfo):
            hfo_s = ch_hfo_dict['start_s'][hfo_idx]
            hfo_e = ch_hfo_dict['end_s'][hfo_idx]
            for ied_idx in range(nr_ied):
                ied_s = ch_ied_dict['start_s'][ied_idx]
                ied_e = ch_ied_dict['end_s'][ied_idx]
                coincident_a = hfo_s >= ied_s and hfo_s <= ied_e
                coincident_b = hfo_e >= ied_s and hfo_e <= ied_e
                coincident = coincident_a or coincident_b
                if coincident:
                    isol_hfo_sel[hfo_idx] = False
                    break

        hfo_rate = np.sum(isol_hfo_sel)/duration_s
        isol_hfo_rates.append(hfo_rate)

    return np.array(channels), np.array(isol_hfo_rates)


def load_gs_file(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    detections_data = mat_contents['detections']

    # store data as a dictionary
    marks = {}
    marks['channel'] = np.array(
        [detection[0][0].lower() for detection in detections_data])

    marks['type'] = np.array([detection[1][0]
                             for detection in detections_data])

    marks['start_s'] = np.array([detection[2][0][0]
                                for detection in detections_data])

    marks['end_s'] = np.array([detection[3][0][0]
                              for detection in detections_data])

    marks['start'] = np.array([detection[4][0][0]
                               for detection in detections_data])

    marks['end'] = np.array([detection[5][0][0]
                            for detection in detections_data])

    marks['comments'] = np.array([detection[6][0]
                                  for detection in detections_data])

    marks['chann_spec'] = np.array(
        [detection[7][0][0] for detection in detections_data])

    marks['creation_t'] = np.array(
        [detection[8][0] for detection in detections_data])

    # marks['username'] = np.array([detection[9][0]
    #                              for detection in detections_data])

    keys_list = list(marks.keys())
    marks_data_shape = marks[keys_list[0]].shape
    for key in marks:
        key_vec_shape = marks[key].shape
        if (len(key_vec_shape) > 1) or (key_vec_shape[0] != marks_data_shape[0]):
            raise Exception(
                "Read featues from marks don't have uniform dimensions")

    return marks


def get_edf_filename(annot_fn, first_name):

    fid_idxs = [i for i in range(len(annot_fn))
                if annot_fn.startswith("-", i)]
    fid_str = annot_fn[fid_idxs[1]+1:fid_idxs[2]]
    edf_files = np.array(get_edf_files())
    edf_sel_fid = [fn.find(fid_str) > 0 for fn in edf_files]
    edf_sel_firstname = [fn.lower().find(first_name) >= 0 for fn in edf_files]
    edf_filename = edf_files[np.logical_and(edf_sel_fid, edf_sel_firstname)]
    return edf_filename


def clean_filename_sep(filename):
    filename = filename.replace("~", "_")
    filename = filename.replace("-", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace("__", "_")
    filename = filename.replace("___", "_")
    filename = filename.replace("____", "_")

    first_name = filename[0:filename.find('_')].lower()
    filename = filename[filename.find('_')+1:-1]
    last_name = filename[0:filename.find('_')].lower()

    return filename, first_name, last_name


def get_valid_scalp_channels(channs_list):

    channs_list = np.array(channs_list)
    valid_sel = np.full(len(channs_list), False)

    for chi in range(len(channs_list)):
        chann = channs_list[chi]
        for mtg in SCALP_LONG_BIP:
            if chann in mtg:
                valid_sel[chi] = True
                break

    return channs_list[valid_sel]
