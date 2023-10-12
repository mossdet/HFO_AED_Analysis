import numpy as np
import mne as mne
import matplotlib.pyplot as plt
import scipy
from get_paths import *
from data_extraction import *
from file_groups import *
from plotters import *


groups = ["All_Patients", "EEG_Improvement",
          "Clinical_Improvement", "Seizure_Free"]

for group_label in groups:

    if group_label == "All_Patients":
        pre_files_list = PRE_FILES
        post_files_list = POST_FILES
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
        eeg_duration_s = eeg_data_pre.times[-1]/60
        hfo_rates_pre = get_hfo_rates(annots_pre, eeg_duration_s)

        annots_post = load_gs_file(post_fn)
        post_edf_fn = edfs_path + post_edf_fn[0]
        eeg_data_post = mne.io.read_raw_edf(post_edf_fn)
        eeg_duration_s = eeg_data_post.times[-1]/60
        hfo_rates_post = get_hfo_rates(annots_post, eeg_duration_s)

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
