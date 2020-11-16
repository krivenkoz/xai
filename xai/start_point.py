import numpy as np
import os
import image_utils as iu

path_to_txt = 'd:\\Krivenko\\XAI\\Work\\2020\\New20201005\\Differences\\Coider\\Differences\\MAD=25\\'
txt_name = 'baboon_blue_diff.txt'
bin_name = txt_name[:-4] + '.bin'

raw_diff_array = iu.convert_difftxt_to_numpy(os.path.join(path_to_txt, txt_name))
raw_diff_array = raw_diff_array + 25
iu.convert_numpy_to_binary(raw_diff_array, datatype='uint8', bin_name=os.path.join(path_to_txt, bin_name))
#iu.plot_numpy2D(raw_diff_array) 