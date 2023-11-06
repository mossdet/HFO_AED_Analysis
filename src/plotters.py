import numpy as np
import scipy
import mne as mne
import matplotlib.pyplot as plt

from get_paths import *
from data_extraction import *
from file_groups import *


def plot_analysis_results(group_label, all_rates_pre, all_rates_post, trim_ok):
    # Plot Results
    nr_pats = int(len(np.unique(all_rates_pre[0, :])))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    sup_title_str = group_label.replace(
        '_', ' ') + "\n" + f"{nr_pats} Patients"
    fig.suptitle(sup_title_str, fontsize=28)

    plot_analysis_boxplot(axs[0], all_rates_pre, all_rates_post, trim_ok)

    plot_analysis_barchart(axs[1], all_rates_pre, all_rates_post, trim_ok)

    plt.tight_layout()
    # wm = plt.get_current_fig_manager()
    # wm.window.state('zoomed')
    plt.show(block=False)

    figFilename = images_path+group_label+'.png'
    plt.savefig(figFilename, bbox_inches='tight', dpi=150)
    plt.close()


def plot_analysis_barchart(ax, all_rates_pre, all_rates_post, trim_ok):

    patids = np.unique(all_rates_pre[0])
    nr_pats = len(patids)
    all_rates = []
    pats_avg_pre = []
    pats_avg_post = []
    nr_pats_same = 0
    nr_pats_decreased = 0
    nr_pats_increased = 0
    for pid in patids:
        pat_sel = all_rates_pre[0, :] == pid
        pre_data = all_rates_pre[1, pat_sel]

        pat_sel = all_rates_post[0, :] == pid
        post_data = all_rates_post[1, pat_sel]

        if trim_ok:
            lp = 5
            hp = 95
            # Trim out outliers
            trim_sel = np.logical_and(pre_data > np.percentile(
                pre_data, lp), pre_data < np.percentile(pre_data, hp))
            pre_data = pre_data[trim_sel]

            trim_sel = np.logical_and(post_data > np.percentile(
                post_data, lp), post_data < np.percentile(post_data, hp))
            post_data = post_data[trim_sel]

        all_rates = np.hstack((all_rates, pre_data, post_data))
        pats_avg_pre.append(np.mean(pre_data))
        pats_avg_post.append(np.mean(post_data))

        use_paired_test = False
        if use_paired_test:
            # zero_method{“wilcox”, “pratt”, “zsplit”
            zero_method_str = 'wilcox'
            stats, p_same = scipy.stats.wilcoxon(
                pre_data, post_data, zero_method=zero_method_str, alternative='two-sided')
            stats, p_decrease = scipy.stats.wilcoxon(
                pre_data, post_data, zero_method=zero_method_str, alternative='greater')
            stats, p_increase = scipy.stats.wilcoxon(
                pre_data, post_data, zero_method=zero_method_str, alternative='less')
        else:
            stats, p_same = scipy.stats.mannwhitneyu(
                pre_data, post_data, alternative='two-sided')
            stats, p_decrease = scipy.stats.mannwhitneyu(
                pre_data, post_data, alternative='greater')
            stats, p_increase = scipy.stats.mannwhitneyu(
                pre_data, post_data, alternative='less')

        alpha_val = 0.0167  # 0.0125
        no_change_s = p_same > alpha_val
        decrease_s = not (
            no_change_s) and p_decrease <= alpha_val and p_decrease < p_same and p_decrease < p_increase
        increase_s = not (
            no_change_s) and p_increase <= alpha_val and p_increase < p_same and p_increase < p_decrease

        if no_change_s+decrease_s+increase_s > 1:
            stop = 1

        nr_pats_same += no_change_s
        nr_pats_decreased += decrease_s
        nr_pats_increased += increase_s

    for idx in range(len(pats_avg_pre)):
        ax.plot([1, 2], [pats_avg_pre[idx], pats_avg_post[idx]], '-k')

    ax.bar(np.zeros(len(pats_avg_pre))+1, pats_avg_pre)
    ax.bar(np.zeros(len(pats_avg_post))+2, pats_avg_post)

    ax.set_title("Patient Analysis", fontsize=20)
    ax.set_xticks([1, 2], ["Pre", "Post"], fontsize=18)
    ax.set_ylabel("HFO/minute", fontsize=18)
    ax.set_ylim(np.min(all_rates), np.max(all_rates))

    nr_pats_same = nr_pats_same/nr_pats*100
    nr_pats_increased = nr_pats_increased/nr_pats*100
    nr_pats_decreased = nr_pats_decreased/nr_pats*100
    textstr = f"Unchanged in {nr_pats_same:.1f}% of patients \n"
    textstr += f"Increase in {nr_pats_increased:.1f}% of patients \n"
    textstr += f"Decrease in {nr_pats_decreased:.1f}% of patients \n"
    textstr += f"(p<{alpha_val})"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.50, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', verticalalignment='top', multialignment='center',  bbox=props)


def plot_analysis_boxplot(ax, all_rates_pre, all_rates_post, trim_ok):

    if trim_ok:
        lp = 5
        hp = 95
        # Trim out outliers
        trim_sel = np.logical_and(all_rates_pre[1, :] > np.percentile(
            all_rates_pre[1, :], lp), all_rates_pre[1, :] < np.percentile(all_rates_pre[1, :], hp))
        all_rates_pre = all_rates_pre[:, trim_sel]

        trim_sel = np.logical_and(all_rates_post[1, :] > np.percentile(
            all_rates_post[1, :], lp), all_rates_post[1, :] < np.percentile(all_rates_post[1, :], hp))
        all_rates_post = all_rates_post[:, trim_sel]

    # Plot Boxplot
    all_rates = np.hstack((all_rates_pre[1, :], all_rates_post[1, :]))
    bplot1 = ax.boxplot((all_rates_pre[1, :], all_rates_post[1, :]),
                        notch=True,  # notch shape
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        )  # will be used to label x-ticks

    ax.set_title("Group Analysis", fontsize=20)
    ax.set_xticks([1, 2], ["Pre", "Post"], fontsize=18)
    ax.set_ylabel("HFO/minute", fontsize=18)
    ax.set_ylim(np.min(all_rates), np.max(all_rates))

    # Perform statisitical tests, ranksums, mannwhitneyu
    stats, p_same = scipy.stats.ranksums(
        all_rates_pre[1, :], all_rates_post[1, :], alternative='two-sided')  # 'two-sided', 'less', 'greater'

    stats, p_decrease = scipy.stats.ranksums(
        all_rates_pre[1, :], all_rates_post[1, :], alternative='greater')  # 'two-sided', 'less', 'greater'

    stats, p_increase = scipy.stats.ranksums(
        all_rates_pre[1, :], all_rates_post[1, :], alternative='less')  # 'two-sided', 'less', 'greater'

    alpha_val = 0.0167  # 0.0125
    no_change_s = p_same > alpha_val
    decrease_s = not (
        no_change_s) and p_decrease <= alpha_val and p_decrease < p_same and p_decrease < p_increase
    increase_s = not (
        no_change_s) and p_increase <= alpha_val and p_increase < p_same and p_increase < p_decrease

    textstr = "HFO Rates pre- and post-AED are the same\n"
    if no_change_s:
        textstr = "HFO Rates pre- and post-AED are the same\n"
    elif decrease_s:
        textstr = "HFO Rates pre- and post-AED are different\n"
        textstr += "HFO Rates decreased after AED\n"
    elif increase_s:
        textstr = "HFO Rates pre- and post-AED are different\n"
        textstr += "HFO Rates increased after AED\n"
    else:
        textstr = "Undetermined behaviour!"

    textstr += f"(p<{alpha_val})"

    # place a text box in upper left in axes coords
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', verticalalignment='top',
            multialignment='center',  bbox=props)


def plot_montage(eeg_data):
    eeg_data = eeg_data.copy().pick_types(meg=False, eeg=True, eog=False)
    valid_channs = get_valid_scalp_channels(eeg_data.ch_names)
    eeg_data.pick_channels(valid_channs)
    info = eeg_data.info
    # easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    # easycap_montage.plot()  # 2D
    # fig = easycap_montage.plot(kind="3d", show=False)  # 3D
    # fig = fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial
    eeg_data.set_montage("standard_1020", on_missing='ignore')
    fig = eeg_data.plot_sensors(show_names=True)

    stop = 1
