##############################################################
# To move soundtracks to a unic directory for calculating 
# metrics such as FAD for background music generation models
#
# python3 move_soundtrack.py --in-folder path/to/original/mtg-jamendo --out-folder /where/to/strore/th/soudtracks --verbose
#
##############################################################

import os
import argparse
from time import perf_counter
import csv

class MoveSoundTrack:
    def __init__(self, in_folder:str, out_folder:str, verbose:bool) -> None:
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.verbose = verbose

    def read_tsv(self, fn):
        r = []
        with open(fn) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            for row in reader:
                r.append(row)
        return r[1:]

    def move(self, split_path, type_):
        tsv_fn = os.path.join(split_path, 'autotagging-%s.tsv'%type_)
        rows = self.read_tsv(tsv_fn)

        count = 0
        for row in rows:
            track_path = os.path.join(self.in_folder, row[3])

            track_name = row[3].split('/')[1]
            track_new_path = os.path.join(self.out_folder, track_name)

            track_tags = row[5:]

            if not os.path.isfile(track_path):
                print(f"{track_path} NOT FOUND")
                continue

            if 'genre---soundtrack' in track_tags:
                count += 1
                if self.verbose:
                    print(f"\nMoving \n  {track_path} \n  with tags \n\t{track_tags}\n  to \n\t{track_new_path}")
                os.rename(track_path, track_new_path)

        return count

    def run(self):
        split_path = '../data/splits/split-0/'
        types = ['train', 'validation', 'test']

        start_time = perf_counter()

        total = 0
        for type in types:
            total += self.move(split_path, type)

        print(f"\nTOTAL: {total}")
        if self.verbose:
            end_time = perf_counter()
            print(f"\n It took {end_time-start_time}s to run")

def _parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-folder', type=str, default='.')
    parser.add_argument('--out-folder', type=str, default='./soudtrack')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args.in_folder, args.out_folder, args.verbose

if __name__ == '__main__':
    in_folder, out_folder, verbose = _parser()

    m = MoveSoundTrack(in_folder, out_folder, verbose)
    m.run()
