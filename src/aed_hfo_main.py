import numpy as np
import pandas as pd

import mne as mne
import matplotlib.pyplot as plt
import scipy
from get_paths import *
from data_extraction import *
from plotters import *
from montage_conversion import *


groups = ["All_Patients", "EEG_Improvement",
          "Clinical_Improvement", "Seizure_Free"]

files_map_fn = "F:/Postdoc_Calgary/Research/AED_and_Scalp_HFO/HFO_AED_Analysis/Data/AED_Patient_Files_Map.xlsx"
files_map = pd.read_excel(files_map_fn)
# pat_info_table_fn = "F:/Postdoc_Calgary/Research/AED_and_Scalp_HFO/HFO_AED_Analysis/Data/AED_Outcome_Table.xlsx"
# pat_info_table = pd.read_csv(pat_info_table_fn)

# groups = ["Seizure_Free"]

for group_label in groups:

    if group_label == "All_Patients":
        pre_files_list = files_map['EDF_Filename'][files_map['Pre'] > 0]
        post_files_list = files_map['EDF_Filename'][files_map['Pre'] <= 0]
    elif group_label == "EEG_Improvement":
        pre_files_list = PRE_FILES_EEG_IMPROVED
        post_files_list = POST_FILES_EEG_IMPROVED
    elif group_label == "Clinical_Improvement":
        pre_files_list = PRE_FILES_CLINICAL_IMPROVED
        post_files_list = POST_FILES_CLINICAL_IMPROVED
    elif group_label == "Seizure_Free":
        pre_files_list = PRE_FILES_SEIZURE_FREE
        post_files_list = POST_FILES_SEIZURE_FREE

    if len(pre_files_list) != len(post_files_list):
        raise Exception("Wrong File Groups!")

    all_rates_pre = []
    all_rates_post = []
    for idx1 in range(len(pre_files_list)):

        pre_fn = []
        post_fn = []

        pre_fn = pre_files_list[idx1]

        cf1, first_name1, last_name1 = clean_filename_sep(pre_fn)

        for idx2 in range(len(post_files_list)):
            cf2, first_name2, last_name2 = clean_filename_sep(
                post_files_list[idx2])
            if first_name1 == first_name2:
                post_fn = post_files_list[idx2]

                pre_edf_fn = get_edf_filename(pre_fn, first_name1)
                post_edf_fn = get_edf_filename(post_fn, first_name2)

                print(pre_fn)
                print(pre_edf_fn)

                print(post_fn)
                print(post_edf_fn)
                print("\n")
                break

        if len(pre_edf_fn) == 0 or len(post_edf_fn) == 0:
            print("EDF not found")

        print(idx1)
        print(pre_edf_fn[0])
        print(post_edf_fn[0]+"\n")

        annots_pre = load_gs_file(pre_fn)
        pre_edf_fn = edfs_path + pre_edf_fn[0]

        eeg_data_pre = mne.io.read_raw_edf(pre_edf_fn)
        # scalp_bp_eeg_pre = convert_to_long_bipolar(eeg_data_pre)
        eeg_duration_s = eeg_data_pre.times[-1]/60
        pre_channels, hfo_rates_pre = get_hfo_rates(annots_pre, eeg_duration_s)

        annots_post = load_gs_file(post_fn)
        post_edf_fn = edfs_path + post_edf_fn[0]
        eeg_data_post = mne.io.read_raw_edf(post_edf_fn)
        # scalp_bp_eeg_post = convert_to_long_bipolar(eeg_data_pre)
        eeg_duration_s = eeg_data_post.times[-1]/60
        post_channels, hfo_rates_post = get_hfo_rates(
            annots_post, eeg_duration_s)

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
            pre_patid = np.full(len(hfo_rates_pre), idx1)
            post_patid = np.full(len(hfo_rates_post), idx1)
            all_rates_pre = np.array((pre_patid, hfo_rates_pre))
            all_rates_post = np.array((post_patid, hfo_rates_post))
        else:
            pre_patid = np.full(len(hfo_rates_pre), idx1)
            post_patid = np.full(len(hfo_rates_post), idx1)
            all_rates_pre = np.hstack(
                (all_rates_pre, np.array((pre_patid, hfo_rates_pre))))
            all_rates_post = np.hstack(
                (all_rates_post, np.array((post_patid, hfo_rates_post))))

    print([np.median(all_rates_pre[1, :]), np.median(all_rates_post[1, :])])

    # Plot Results
    plot_analysis_results(group_label, all_rates_pre,
                          all_rates_post, trim_ok=False)


stop = 1
