# PaSST on the MTG-Jamendo Dataset
This is a fork from the [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset) that aims on fine-tuning [PaSST](https://github.com/kkoutini/passt_hear21) on it.

## Preprocessing
### Converting the audios
From the [scripts/passt](https://github.com/FelipeMarra/passt-on-mtg-jamendo-dataset/tree/master/scripts/passt) run [preprocess.py](https://github.com/FelipeMarra/passt-on-mtg-jamendo-dataset/blob/master/scripts/passt/preprocess.py):
```bash
python3 src/preprocess.py --in_folder /path/to/mtg/jamendo/dataset/audio/folders --out-folder /where/to/store/the/preprocessed/files
```
This will, using all your CPU threads, convert the original audios into 32KHz, mono, 10s, WAV segments. Using the `--remove` flag will remove the original dataset files as soon as they get processed.

### Getting the dictionaries used by the DataLoader
From the [scripts/passt](https://github.com/FelipeMarra/passt-on-mtg-jamendo-dataset/tree/master/scripts/passt) run [get_dicts.py](https://github.com/FelipeMarra/passt-on-mtg-jamendo-dataset/blob/master/scripts/passt/get_dicts.py):
```bash
python3 get_dicts.py --in-folder /where/you/stored/the/preprocessed/files
```
This is a modified version of the original code from the MTG dataset. Here I check if the audios exist instead of the mel spectrograms. If the audios don't exist they will be skipped, allowing for training on a subset. The script will creat dicts in .picke format inside the splits folders that will be used by the DataLoader. Their keys are a index, from 0 to N, where N is the number of audio segments. Their values store another dicts, containing the path to the audio in the format `folder_inside_the_dataset/track_id` and the tags for that track in one-hot encoding.

## Trainig
Run [scripts/passt/main.py](https://github.com/FelipeMarra/passt-on-mtg-jamendo-dataset/blob/master/scripts/passt/main.py):
```bash
python3 main.py --in-folder /where/you/stored/the/preprocessed/files --batch-size B
```
