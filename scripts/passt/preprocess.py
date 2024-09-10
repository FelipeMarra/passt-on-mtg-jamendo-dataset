import os
import math
import argparse
import multiprocessing as mp
from time import perf_counter

import ffmpeg

def process_audios(in_folder:str, in_files:list[str], out_folder:str):
    for file in in_files:
        in_file = os.path.join(in_folder, file)

        out_file = in_file.split('/')[-1].split('.')[0]+'_%03d.wav'
        out_file = os.path.join(out_folder, out_file)

        if os.path.isfile(out_file.split('%')[0]+'000.wav'):
            continue

        # Convert to mono 32khz wav and split in 10s segments
        stream = ffmpeg.input(in_file)
        stream = ffmpeg.output(stream, out_file, f='segment', segment_time=10, ar=32000, ac=1)
        ffmpeg.run(stream)

        os.remove(in_file)

def _parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-folder', type=str, default='.')
    parser.add_argument('--out-folder', type=str, default='.')

    args = parser.parse_args()
    return args.in_folder, args.out_folder

if __name__ == "__main__":
    in_folder, out_folder = _parser()

    threads = mp.cpu_count()
    print(f"\n Got {threads} threads \n")

    files = os.listdir(in_folder)
    dataset_split = math.ceil(len(files)/threads)

    pool = mp.Pool()
    start_time = perf_counter()

    for t in range(threads):
        start = t*dataset_split
        end = dataset_split*(t+1)
        end = end if end <= len(files) else len(files)

        pool.apply_async(
            process_audios, args=(in_folder,
                                files[start:end], 
                                out_folder)
        )

    pool.close()
    pool.join()

    end_time = perf_counter()
    print(f"\n It took {end_time-start_time}s")