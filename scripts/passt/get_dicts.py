import os
import argparse
from time import perf_counter
import multiprocessing as mp

import csv
import pickle
import numpy as np

class Split:
    def __init__(self, in_folder:str, verbose:bool) -> None:
        self.in_folder = in_folder
        self.verbose = verbose

        self.folders:dict[str,list[str]] = {}

        for folder_idx in os.listdir(self.in_folder):
            folder = os.path.join(self.in_folder, folder_idx)
            folder_files = sorted(os.listdir(folder))
            self.folders[folder_idx] = folder_files

    def read_tsv(self, fn):
        r = []
        with open(fn) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            for row in reader:
                r.append(row)
        return r[1:]

    def get_tag_list(self, option):
        if option == 'top50tags':
            if not os.path.isfile('tag_list_50.npy'):
                print("Unnable to load top50tags. Run the script from its containing folder.")

            tag_list = np.load('tag_list_50.npy')
        else:
            if not os.path.isfile('tag_list.npy'):
                print("Unnable to load top50tags. Run the script from its containing folder.")

            tag_list = np.load('tag_list.npy')

            if option == 'genre':
                tag_list = tag_list[:87]
            elif option == 'instrument':
                tag_list = tag_list[87:127]
            elif option == 'moodtheme':
                tag_list = tag_list[127:]

        return list(tag_list)

    def get_slices_list(self, track:str):
        # track is in the form folder/track_id
        track = track.split('/')
        track_folder = track[0]
        track_file = track[1]

        # look for the first incidence of the track id
        folder_files = self.folders.get(track_folder)

        if folder_files == None:
            #print(f"Unnable to find folder {path_folder}")
            return []

        query = track_file[:-4]+'_'

        fst_incidence = 0

        for idx, file in enumerate(folder_files):
            if file.startswith(query):
                fst_incidence = idx
                break

        # append to slices_list until there is no other incidence
        slices_list = []
        idx = fst_incidence

        while idx < len(folder_files) and folder_files[idx].startswith(query):
            slice_path = os.path.join(track_folder, folder_files[idx])

            slices_list.append(slice_path)
            idx += 1

        if self.verbose:
            print(f"\n The track {track_file} has the following {len(slices_list)} incidences: {slices_list}\n")

        return slices_list

    def get_dict(self, split_path, tag_list, option, type_='train'):
        if option=='all':
            tsv_fn = os.path.join(split_path, 'autotagging-%s.tsv'%type_)
        else:
            tsv_fn = os.path.join(split_path, 'autotagging_%s-%s.tsv'%(option, type_))
        rows = self.read_tsv(tsv_fn)

        dictionary = {}
        i = 0
        for row in rows:
            track_path = row[3]
            slices_list = self.get_slices_list(track_path)

            for slice_path in slices_list:
                temp_dict = {}
                temp_dict['path'] = slice_path
                temp_dict['duration'] = (float(row[4]) * 12000 - 512) // 256

                if option == 'all':
                    temp_dict['tags'] = np.zeros(183)
                elif option == 'genre':
                    temp_dict['tags'] = np.zeros(87)
                elif option == 'instrument':
                    temp_dict['tags'] = np.zeros(40)
                elif option == 'moodtheme':
                    temp_dict['tags'] = np.zeros(56)
                elif option == 'top50tags':
                    temp_dict['tags'] = np.zeros(50)

                tags = row[5:]
                for tag in tags:
                    try:
                        temp_dict['tags'][tag_list.index(tag)] = 1
                    except:
                        continue

                if temp_dict['tags'].sum() > 0:
                    dictionary[i] = temp_dict
                    i += 1

        dict_fn = os.path.join(split_path, '%s_%s_dict.pickle'%(option, type_))

        with open(dict_fn, 'wb') as pf:
            pickle.dump(dictionary, pf)

    def run_iter(self, split, option='all'):
        tag_list = self.get_tag_list(option)
        split_path = '../../data/splits/split-%d/' % split
        types = ['train', 'validation', 'test']

        pool = mp.Pool()
        start_time = perf_counter()

        for type in types:
            pool.apply_async(
                self.get_dict, args=(split_path, tag_list, option, type)
            )

        pool.close()
        pool.join()

        if self.verbose:
            end_time = perf_counter()
            print(f"\n It took {end_time-start_time}s to run {option} on split {split}")

    def run(self):
        options = ['all', 'genre', 'instrument', 'moodtheme', 'top50tags']

        start_time = perf_counter()

        for i in range(5):
            for option in options:
                self.run_iter(i, option)

        end_time = perf_counter()
        print(f"\n It took {end_time-start_time}s to run it all")

def _parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-folder', type=str, default='.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args.in_folder, args.verbose

if __name__ == '__main__':
    in_folder, verbose = _parser()

    s = Split(in_folder, verbose)
    s.run()