import numpy as np
import pandas as pd

import mne as mne
import matplotlib.pyplot as plt
import scipy
from get_paths import *
from data_extraction import *
from plotters import *
from montage_conversion import *

hfo_types = ["HFO", "iesHFO", "isolHFO"]
groups = ["All_Patients", "EEG_Improvement",
          "Clinical_Improvement", "Seizure_Free"]

files_map_fn = "F:/Postdoc_Calgary/Research/AED_and_Scalp_HFO/HFO_AED_Analysis/Data/AED_Patient_Files_Map.xlsx"
files_map = pd.read_excel(files_map_fn)
# pat_info_table_fn = "F:/Postdoc_Calgary/Research/AED_and_Scalp_HFO/HFO_AED_Analysis/Data/AED_Outcome_Table.xlsx"
# pat_info_table = pd.read_csv(pat_info_table_fn)

# groups = ["Seizure_Free"]

for hfo_type in hfo_types:
    for group_label in groups:

        pre_sel = files_map['Pre'] > 0
        post_sel = np.logical_not(pre_sel)
        if group_label != "All_Patients":
            pre_sel = np.logical_and(pre_sel, files_map[group_label] > 0)
            post_sel = np.logical_and(post_sel, files_map[group_label] > 0)

        pre_annfiles_list = np.array(
            files_map['Annotations_Filename'])[pre_sel]
        pre_edffiles_list = np.array(files_map['EDF_Filename'])[pre_sel]
        post_annfiles_list = np.array(
            files_map['Annotations_Filename'])[post_sel]
        post_edffiles_list = np.array(files_map['EDF_Filename'])[post_sel]

        nr_pats = len(pre_annfiles_list)

        if (len(pre_annfiles_list)+len(post_annfiles_list)+len(pre_edffiles_list)+len(post_edffiles_list))/4 != nr_pats:
            raise Exception("Wrong File Groups!")

        all_rates_pre = []
        all_rates_post = []
        for fidx in range(len(pre_annfiles_list)):

            pre_annots_fn = annotations_path+pre_annfiles_list[fidx]
            pre_edf_fn = edfs_path+pre_edffiles_list[fidx]+'.edf'
            post_annots_fn = annotations_path+post_annfiles_list[fidx]
            post_edf_fn = edfs_path+post_edffiles_list[fidx]+'.edf'

            # Pre-rates
            annots_pre = load_gs_file(pre_annots_fn)
            eeg_data_pre = mne.io.read_raw_edf(pre_edf_fn)
            # scalp_bp_eeg_pre = convert_to_long_bipolar(eeg_data_pre)
            eeg_duration_pre_s = eeg_data_pre.times[-1]/60
            if hfo_type == 'HFO':
                pre_channels, hfo_rates_pre = get_hfo_rates(
                    annots_pre, eeg_duration_pre_s)
            elif hfo_type == 'iesHFO':
                pre_channels, hfo_rates_pre = get_ieshfo_rates(
                    annots_pre, eeg_duration_pre_s)
            elif hfo_type == 'isolHFO':
                pre_channels, hfo_rates_pre = get_isolhfo_rates(
                    annots_pre, eeg_duration_pre_s)

            # Post-rates
            annots_post = load_gs_file(post_annots_fn)
            eeg_data_post = mne.io.read_raw_edf(post_edf_fn)
            # scalp_bp_eeg_post = convert_to_long_bipolar(eeg_data_pre)
            eeg_duration_post_s = eeg_data_post.times[-1]/60
            if hfo_type == 'HFO':
                post_channels, hfo_rates_post = get_hfo_rates(
                    annots_post, eeg_duration_post_s)
            elif hfo_type == 'iesHFO':
                post_channels, hfo_rates_post = get_ieshfo_rates(
                    annots_post, eeg_duration_post_s)
            elif hfo_type == 'isolHFO':
                post_channels, hfo_rates_post = get_isolhfo_rates(
                    annots_post, eeg_duration_post_s)

            date_diff = eeg_data_post.info['meas_date'] - \
                eeg_data_pre.info['meas_date']

            assert (date_diff.total_seconds() > 0,
                    "Pre and Post Files are mixed up")

            consider_only_shared_channels = False
            if consider_only_shared_channels:
                # keep only channels present in both pre and post
                keep_pre = np.full(len(pre_channels), False)
                for chfi in range(len(pre_channels)):
                    keep_pre[chfi] = pre_channels[chfi] in post_channels
                hfo_rates_pre = hfo_rates_pre[keep_pre]
                pre_channels = pre_channels[keep_pre]
                # sort channels alphabetically
                hfo_rates_pre = hfo_rates_pre[np.argsort(pre_channels)]
                pre_channels = pre_channels[np.argsort(pre_channels)]

                keep_post = np.full(len(post_channels), False)
                for chfi in range(len(post_channels)):
                    keep_post[chfi] = post_channels[chfi] in pre_channels
                hfo_rates_post = hfo_rates_post[keep_post]
                post_channels = post_channels[keep_post]
                # sort channels alphabetically
                hfo_rates_post = hfo_rates_post[np.argsort(post_channels)]
                post_channels = post_channels[np.argsort(post_channels)]

                for chfi in range(len(pre_channels)):
                    if pre_channels[chfi] != post_channels[chfi]:
                        raise Exception("Pre and Post Channels not matching!")

                if len(keep_pre) != sum(keep_pre):
                    stop = 1
                if len(keep_post) != sum(keep_post):
                    stop = 1

            # Trim outlier channels
            trim_ok = False
            if trim_ok:
                lp = 10
                hp = 90
                # Trim out outliers
                hfo_rates_pre = np.array(hfo_rates_pre)
                trim_sel = np.logical_and(hfo_rates_pre > np.percentile(
                    hfo_rates_pre, lp), hfo_rates_pre < np.percentile(hfo_rates_pre, hp))
                hfo_rates_pre = hfo_rates_pre[trim_sel].tolist()

                hfo_rates_post = np.array(hfo_rates_post)
                trim_sel = np.logical_and(hfo_rates_post > np.percentile(
                    hfo_rates_post, lp), hfo_rates_post < np.percentile(hfo_rates_post, hp))
                hfo_rates_post = hfo_rates_post[trim_sel].tolist()

            if len(all_rates_pre) == 0:
                pre_patid = np.full(len(hfo_rates_pre), fidx)
                post_patid = np.full(len(hfo_rates_post), fidx)
                all_rates_pre = np.array((pre_patid, hfo_rates_pre))
                all_rates_post = np.array((post_patid, hfo_rates_post))
            else:
                pre_patid = np.full(len(hfo_rates_pre), fidx)
                post_patid = np.full(len(hfo_rates_post), fidx)
                all_rates_pre = np.hstack(
                    (all_rates_pre, np.array((pre_patid, hfo_rates_pre))))
                all_rates_post = np.hstack(
                    (all_rates_post, np.array((post_patid, hfo_rates_post))))

        print([np.median(all_rates_pre[1, :]), np.median(all_rates_post[1, :])])

        # Plot Results
        plot_analysis_results(group_label, all_rates_pre,
                              all_rates_post, hfo_type, trim_ok=False)


pass
