import numpy as np
import pandas as pd
import os
import image_utils as iu

# Test difference
if 0:
    path_to_txt = 'd:\\Krivenko\\XAI\\Work\\2020\\New20201005\\Differences\\Coider\\Differences\\MAD=25\\'
    txt_name = 'baboon_blue_diff.txt'
    bin_name = txt_name[:-4] + '.bin'

    raw_diff_array = iu.convert_difftxt_to_numpy(os.path.join(path_to_txt, txt_name))
    #raw_diff_array = raw_diff_array + 25
    iu.convert_numpy_to_binary(raw_diff_array, datatype='uint8', bin_name=os.path.join(path_to_txt, bin_name))
    iu.plot_numpy2D(raw_diff_array)

# Calc data for all differences
if 1:
    images = ['baboon', 'lena', 'monarch']
    colors = ['red', 'green', 'blue']
    maxdiff = [9, 25]
    path_to_txt = 'd:\\Krivenko\\XAI\\Work\\2020\\New20201005\\Differences\\Coider\\Differences\\MAD='
    path_to_coder = 'd:\\Krivenko\\XAI\\Work\\2020\\New20201005\\Differences\\Coider\\'
    df = pd.DataFrame(columns=['Image','Color','Max difference','Binary size','ZLIB file size',
                                'compr05 file size','CR ZLIB', 'CR compr05'])

    i = 0    
    for diffe in maxdiff:
        for image in images:                    
            for color in colors:
                full_path = path_to_txt + str(diffe) + '\\' + image + '_' + color + '_diff.txt'
                iname = image + '_' + color + '_' + str(diffe)
                raw_diff_array = iu.convert_difftxt_to_numpy(full_path)
                bin_name = iname + '.bin'
                iu.convert_numpy_to_binary(raw_diff_array, datatype='int8', bin_name=bin_name)
                bin_name_size, zlib_name_size, CR_zlib = iu.compress_zlib(iname + '.bin', iname + '.zlib')
                bin_name_size, nadia_name_size, CR_nadia = iu.compress_nadia(path_to_coder, iname + '.bin', iname + '.nadia')
                os.remove(iname + '.bin')

                df.loc[i, 'Image'] = image
                df.loc[i, 'Color'] = color
                df.loc[i, 'Max difference'] = diffe
                df.loc[i, 'Binary size'] = bin_name_size
                df.loc[i, 'ZLIB file size'] = zlib_name_size
                df.loc[i, 'compr05 file size'] = nadia_name_size
                df.loc[i, 'CR ZLIB'] = CR_zlib
                df.loc[i, 'CR compr05'] = CR_nadia
                i += 1
    df.to_csv('Difference_result.csv', index=False)
    df.to_excel('Difference_result.xlsx', sheet_name='Differences')