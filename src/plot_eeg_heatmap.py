import numpy as np
import mne as mne
from get_paths import *
from data_extraction import *


easycap_montage = mne.channels.make_standard_montage("easycap-M1")
print(easycap_montage)
ssvep_folder = mne.datasets.ssvep.data_path()
ssvep_data_raw_path = (
    ssvep_folder / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)
ssvep_raw = mne.io.read_raw_brainvision(ssvep_data_raw_path, verbose=False)

# Use the preloaded montage
ssvep_raw.set_montage(easycap_montage)
fig = ssvep_raw.plot_sensors(show_names=True)

# Apply a template montage directly, without preloading
ssvep_raw.set_montage("easycap-M1")
fig = ssvep_raw.plot_sensors(show_names=True)
