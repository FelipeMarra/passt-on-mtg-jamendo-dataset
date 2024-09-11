import os
import math
import argparse
import multiprocessing as mp
from time import perf_counter

import ffmpeg

def process_audios(file_tuples:list[tuple[str, str]], remove:bool):
    for file_tuple in file_tuples:
        in_file, out_file = file_tuple

        if os.path.isfile(out_file):
            continue

        # Convert to mono 32khz wav and split in 10s segments
        stream = ffmpeg.input(in_file)
        stream = ffmpeg.output(stream, out_file, f='segment', segment_time=10, ar=32000, ac=1)
        ffmpeg.run(stream)

        if remove:
            os.remove(in_file)

def _parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-folder', type=str, default='.')
    parser.add_argument('--out-folder', type=str, default='.')
    parser.add_argument('--remove', action='store_true')

    args = parser.parse_args()
    return args.in_folder, args.out_folder, args.remove

if __name__ == "__main__":
    in_folder, out_folder, remove = _parser()

    threads = mp.cpu_count()
    print(f"\n Got {threads} threads \n")

    all_files = []

    for folder in os.listdir(in_folder):
        in_folder_idx = os.path.join(in_folder, folder)
        out_folder_idx = os.path.join(out_folder, folder)

        if not os.path.isdir(out_folder_idx):
            os.makedirs(out_folder_idx)

        in_folder_idx_list = os.listdir(in_folder_idx)
        for file in in_folder_idx_list:
            in_file = os.path.join(in_folder_idx, file)

            out_file = file.split('.')[0]+'_%03d.wav'
            out_file = os.path.join(out_folder_idx, out_file)

            all_files.append((in_file, out_file))

    print(f"\n Got {len(all_files)} audio tracks \n")
    print(all_files[:5], all_files[-5:])

    dataset_split = math.ceil(len(all_files)/threads)

    pool = mp.Pool()
    start_time = perf_counter()

    for t in range(threads):
        start = t*dataset_split
        end = dataset_split*(t+1)
        end = end if end <= len(all_files) else len(all_files)

        pool.apply_async(
            process_audios, args=(all_files[start:end], remove)
        )

    pool.close()
    pool.join()

    end_time = perf_counter()
    print(f"\n It took {end_time-start_time}s")