#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the fairseq repository.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple
import re

import numpy as np
import pandas as pd
import soundfile as sf
#from examples.speech_to_text.data_utils import (
from examples.basic_speech_to_text.data.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)


import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.basic_speech_to_text.data.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh"]

    def __init__(self, root: str, lang: str, split: str, pair_type: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        num_segments = len(segments)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert num_segments == len(utterances)
            for i, u in enumerate(utterances):
                u = self.edit_utterance(u, _lang, pair_type)
                segments[i][_lang] = u

        if type(pair_type) is not bool and (pair_type is not None) and (pair_type != "none"):
            print(f"Creating paired dataset with method: {pair_type}", flush=True)
            pair_segments = []
            for i, cur_segment in enumerate(segments):
                if i+1 < num_segments:
                    next_segment = segments[i+1]
                else:
                    next_segment = None
                pair_segment = {k: v for k,v in cur_segment.items()}    # Deepcopy in case we want to keep original segments unchanged
                pair_segment["speaker_id"] += "_pair"

                if (pair_type == "partial") or (pair_type == "original+partial") :
                    # 'Partial' method uses 1.5 seconds of audio from another segment, but none of the text
                    # This method aims to make model aware of potential for subsequent sentences
                    # Placing the end of context token <e> therefore requires learning to ignore audio after the current sentence
                    
                    # Only use cur_segment with one sentence, but don't care about next_segment length since we're only using a small amount of audio anyways
                    if (next_segment is not None) and (cur_segment["wav"] == next_segment["wav"]) and (self.get_sentence_count(cur_segment[lang]) == 1) and (self.get_sentence_count(next_segment[lang]) >= 1):
                        pair_segment["en"] = cur_segment["en"]
                        pair_segment[lang] = cur_segment[lang]

                        base_duration = float(next_segment["offset"]) - float(cur_segment["offset"])        # Time before first word of next segment
                        pair_duration = base_duration + min(1.5, float(next_segment["duration"]))           # Add either 1.5 seconds or duration of next segment
                        pair_segment["duration"] = str( pair_duration )

                        pair_segments.append(pair_segment)

                elif (pair_type == "full") or (pair_type == "original+full"):
                    # 'Full' method uses all audio & text from the next sentence/segment
                    # This method aims to make model aware of potential contextual information, or just consistency across sentences
                    # But makes assumption that there will "always" be a 2nd sentence, so could introduce odd behavior if evaluating on single-sentence dataset
                    
                    if (next_segment is not None) and (cur_segment["wav"] == next_segment["wav"]) and (self.get_sentence_count(cur_segment[lang]) == 1) and (self.get_sentence_count(next_segment[lang]) == 1):
                        pair_segment["en"] = cur_segment["en"] + " " + next_segment["en"]
                        pair_segment[lang] = cur_segment[lang] + " " + next_segment[lang]
                        pair_segment["duration"] = str( float(next_segment["offset"]) - float(cur_segment["offset"]) + float(next_segment["duration"]) )

                        pair_segments.append(pair_segment)

                    elif self.get_sentence_count(cur_segment["en"]) == 2:
                        pair_segments.append(pair_segment)

            if "original" in pair_type:
                segments = segments + pair_segments
            else:
                segments = pair_segments

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}_pair" if ("pair" in segment["speaker_id"]) else f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )


    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)

    #Edits the utterance by removing unecessary punctuations and adding special characters
    def edit_utterance(self, utterance, lang, pair_type):

        end_characters = {'en': ['.', '?', '!'], 'de': ['.', '?', '!'], 'es': ['.', '?', '!'], 'zh': ['。', '？', '！']}

        # General punctuation cleanup
        utterance = utterance.replace("(Video)", "").replace("(video)", "").replace("(Audio)", "").replace("(audio)", "")
        utterance = utterance.replace("[", "").replace("]", "").replace("「", "").replace("」", "").replace("｢", "").replace("｣", "")
        utterance = utterance.replace("“", "").replace("”", "").replace("\"", "").replace("＂", "").replace("《", ""). replace("》", "").replace("【", "").replace("】", "").replace("‟", "").replace("„", "")
        utterance = utterance.replace("（", "(").replace("）", ")").replace("：", ":").replace("；", ";").replace("～", "~").replace("％", "%").replace("．", ".").replace("⋯", ".")
        utterance = utterance.replace("’", "'").replace("‘", "'")
        utterance = utterance.replace("—", "–").replace("─", "–")                     # Convert all dash types (0x2014, 0x2500) to regular en dash
        utterance = utterance.replace("－", "-")                                      # Convert all hyphen types (0xFF0D) to regular hyphen
        utterance = utterance.replace("﹖", "?")
        utterance = utterance.replace(" / ", " ")
        utterance = utterance.replace("©", "").replace("®", "").replace("™", "").replace("♪", "").replace("♫", "")

        if len(utterance) == 0:
            return utterance

        if lang in ["zh"]:
            utterance = utterance.replace(" -- ", "--").replace(" --", "--").replace("-- ", "--").replace("--", "，")                     # Convert all hyphen used for pause to commas
            utterance = utterance.replace("~", "–").replace("––", "–")        # Convert all tilde & en dashes to single en dash
            utterance = utterance.replace(", ", "，").replace(",", "，").replace(";", "。").replace("…", "。")

            utterance = ' '.join(utterance.strip().split())
            if (utterance[-1] == '，') or (utterance[-1] == '、') or (utterance[-1] == "–") or (utterance[-1] == "-"):   # Replace end if comma (multiple types) or dash/hyphen
                utterance = ' '.join(utterance[:-1].strip().split()) + '。'
            if (utterance[0] == '，') or (utterance[0] == '、') or (utterance[0] == "–"):
                utterance = utterance[1:]                           # Remove start if comma (multiple types) or dash
            utterance = ' '.join(utterance.strip().split())
        else:
            utterance = utterance.replace("—", ", ").replace("–", ", ").replace(" -- ", ", ").replace("--", ", ")    # Convert all dash & double hyphen to comma
            utterance = utterance.replace(";", ", ")

            utterance = ' '.join(utterance.strip().split())
            if (utterance[-1] == ',') or (utterance[-1] == "-"):   # Replace end if comma or hyphen
                utterance = ' '.join(utterance[:-1].strip().split()) + '.'
            if (utterance[0] == ',')  or (utterance[0] == "-"):
                utterance = utterance[1:]                           # Remove start if comma or hyphen
            utterance = ' '.join(utterance.strip().split())

        #Complicated edits after general punctuation cleanup
        if lang not in ["zh"]:
            utterance = self.fix_and_remove_apostrophes(utterance)
        else:
            utterance = utterance.replace("'", "")

        utterance = self.replace_abbreviations(utterance, lang)
        utterance = self.replace_pauses(utterance, lang)
        utterance = self.replace_non_spoken(utterance, lang)
        utterance = utterance.replace("(", "").replace(")", "")

        utterance = self.remove_speaker_initials(utterance)
        utterance = self.remove_speaker_two_name(utterance, lang)
        utterance = utterance.replace(":", "")

        utterance = self.fix_end_characters(utterance, end_characters, lang)
        utterance = self.remove_repeating_end(utterance, end_characters[lang], lang)

        utterance = ' '.join(utterance.strip().split())     # Clean up white space
        utterance = self.check_no_end(utterance, lang)
        if pair_type != None:
            utterance = self.add_terminator(utterance, end_characters[lang])

        return utterance

    def replace_abbreviations(self, utterance, lang):
        abbreviations = {'en': {'Mr.': 'Mister', 'Ms.': 'Miss', 'Mrs.': 'Missus', 'Dr.': 'Doctor', 'St.': 'Saint', 'Jr.': 'Junior'}}
        if lang in abbreviations.keys():
            for k,v in abbreviations[lang].items():
                utterance = utterance.replace(k, v)
        return utterance

    # Make sure all end characters are correct for the language (remove 'en' characters from other languages)
    def fix_end_characters(self, utterance, end_characters, lang):
        for end_char_idx in list(range(len(end_characters['en']))):
            check_char = end_characters['en'][end_char_idx]
            index = utterance.find(end_characters['en'][end_char_idx])
            while index != -1:
                # If end of sentence, always replace
                if index+1 == len(utterance):
                    utterance = utterance[:index] + end_characters[lang][end_char_idx]
                # Substitute en character unless followed by english character or number
                elif not (utterance[index+1].isascii() and utterance[index+1].isalnum()):
                    utterance = utterance[:index] + end_characters[lang][end_char_idx] + utterance[index+1:]
                index = utterance.find(end_characters['en'][end_char_idx], index+1)
        return utterance


    #Replaces specific noise captions with pause token <0>
    def replace_pauses(self, utterance, lang):
        index_start = utterance.find("(")
        while index_start != -1:
            index_end = utterance.find(")", index_start)
            index_start_next = utterance.find("(", index_start+1)

            if (index_end == -1) or ((index_end > index_start_next) and (index_start_next != -1)):
                # Current "(" has no match, so just remove
                utterance = utterance[:index_start] + utterance[index_start+1:]
                index_start = utterance.find("(", index_start)
            else:
                segment = utterance[index_start:index_end+1]
                if lang == "zh":
                    # If (applause) or (laughter) or (clapping), replace
                    if segment == "(掌声)" or segment == "(笑声)" or segment == "(鼓掌)" or segment == "(笑)":
                        utterance = utterance[:index_start] + "<0>" + utterance[index_end+1:]
                        index_start = utterance.find("(")
                    else:
                        index_start = utterance.find("(", index_start + 1)
                else:
                    #Replaces pause with <0> if first letter is capitalized
                    if utterance[index_start+1].isupper():
                        utterance = utterance[:index_start] + "<0>" + utterance[index_end+1:]
                        index_start = utterance.find("(")
                    else:
                        index_start = utterance.find("(", index_start + 1)

        return utterance

    def fix_and_remove_apostrophes(self, utterance):
        # First loop: fix apostrophes
        #Finds index of first apostrophe
        index = utterance.find("'")
        while index != -1:
            # One condition to fix:
            # apostrophe has char + space before and space + word after, with word after being <= 2 chars (most likely contraction or possession)
            # e.g., They ' re , We ' ve , He ' s
            if (index-2 >=0 and index+3 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1]==' ' and utterance[index+1]==' ' and utterance[index+2].isalpha() and utterance[index+3]==' '):
                utterance = utterance[:index-1] + "'" + utterance[index+2:] # Chop out spaces
            elif (index-2 >=0 and index+4 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1]==' ' and utterance[index+1]==' ' and utterance[index+2].isalpha() and utterance[index+3].isalpha() and utterance[index+4]==' '):
                utterance = utterance[:index-1] + "'" + utterance[index+2:] # Chop out spaces
            index+=1
            index = utterance.find("'", index)

        # Second loop: remove undesired apostrophes
        #Finds index of first '
        index = utterance.find("'")
        while index != -1:
            # Only two allowed conditions:
            # 1) after any alpha + "s" and before space (e.g., childrens' )
            # 2) between two alpha chars (e.g., He's )
            if (index-2 >= 0 and index+1 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1] == 's' and utterance[index+1]==' '):
                index += 1
                pass
            elif (index-1 >= 0 and index+1 < len(utterance)) and \
              (utterance[index-1].isalpha() and utterance[index+1].isalpha()):
                index += 1
                pass
            else:
                utterance = utterance[:index] + utterance[index+1:]
            index = utterance.find("'", index)
        return utterance

    def replace_non_spoken(self, utterance, lang):
        # Remove extra notes made by captioner that are not actually spoken
        if lang in 'zh':
            # Remove any translations by captioner in parentheses
            index_start = utterance.find("(")
            while index_start != -1:
                index_end = utterance.find(")", index_start)
                if index_end == -1:
                    break

                segment = utterance[index_start:index_end+1]
                # isascii() is convenient way to check for all English characters
                if segment.isascii():
                    utterance = utterance[:index_start] + utterance[index_end+1:]

                index_start = utterance.find("(", index_start + 1)
        return utterance

    #Removes the speakers initials if it comes before a sentence
    def remove_speaker_initials(self, utterance):
        #Finds index of first :
        index = utterance.find(":")
        #Conditional satisfied if colon located in utterance AND
        #   entire sub-string before colon is uppercase
        if (index != -1) and (utterance[0:index].isupper()):
            return utterance[index+1:]
        else:
            return utterance

    #Removes the speakers first and last name from beginning of sentence
    def remove_speaker_two_name(self, utterance, lang):
        # Check for names in English
        #Finds index of : and space
        index_colon = utterance.find(":")
        index_space = utterance.find(" ")
        # Condition satisfied if both colon and space present, and space before colon
        if (index_space != -1) and (index_colon != -1) and (index_space < index_colon):
            #Condition satisfied if only one space before colon (i.e., one word) and
            #  word after space starts with upper
            #  and, if space after colon, then not lower after that
            if (utterance[:index_colon].count(" ") == 1) and (utterance[index_space+1].isupper()) and not (index_colon+2<len(utterance) and utterance[index_colon+1]==" " and utterance[index_colon+2].islower()):
                return utterance[index_colon+1:]

        # Check for names in other languages
        if lang in ['zh']:
            # Find index of : and · (middle dot) -- this assumes we converted all colons to :
            index_colon = utterance.find(":")
            index_dot = utterance.find("·")
            # Condition satisfied if both colon and mid dot present and mid dot present colon
            if (index_dot != -1) and (index_colon != -1) and (index_dot < index_colon):
                # Condition satisfied if everything before colon (except dot) is alphabetic or space (except dot) AND
                #    index_colon < 10 (approximate rule to prevent big mistakes)
                if all(x.isalpha() or x.isspace() for x in utterance[0:index_dot]+utterance[index_dot+1:index_colon]) and (index_colon < 10):
                    return utterance[index_colon+1:]
        return utterance

    #Removes a repeating character from the end of sentences
    def remove_repeating_end(self, utterance, end_characters, lang):
        for end_character in end_characters:
            start_idx = end_idx = utterance.find(end_character)
            #Loop executes while there are still repeating end characters
            while utterance.count(end_character + end_character) != 0:
                #Finds index of the character after end_character
                while end_idx < len(utterance) and utterance[end_idx] == end_character:
                    end_idx += 1
                # If not repeating character, no need to check anything so go to next loop
                if end_idx - start_idx == 1:
                    start_idx = end_idx = utterance.find(end_character, start_idx+1)
                    continue

                #Conditional satisfied if not at end of sentence
                if end_idx < len(utterance):
                    if lang in ["en", "de"]:
                        # Conditional satisfied if the following character is a number (e.g., 1.7 billion) or
                        # a space at the end (blank caused by prior processing) or
                        # the following character + 1 is a capital (e.g., "Good ... Let's do that.") or non-period special character
                        if utterance[end_idx].isdigit() or \
                          (utterance[end_idx] == ' ' and end_idx + 1 == len(utterance)) or \
                          (end_idx + 1 < len(utterance) and (utterance[end_idx + 1].isupper() or (not utterance[end_idx + 1].isalnum() and utterance[end_idx + 1] != "."))):
                            utterance = utterance[0:start_idx] + end_character + utterance[end_idx:]
                        else:
                            utterance = utterance[0:start_idx] + utterance[end_idx:]
                    else:
                        utterance = utterance[0:start_idx] + utterance[end_idx:]
                #Conditional satisfied if at end of sentence
                elif end_idx == len(utterance):
                    utterance = utterance[0:start_idx] + end_character

                #Finds the index of next end_character
                start_idx = end_idx = utterance.find(end_character)

        return utterance

    # Checks/adds terminating character if utterance does not already end with terminating character (i.e., sentence)
    def check_no_end(self, utterance, lang):
        if len(utterance) == 0:
            return utterance
        elif utterance == "<0>" or utterance == "<0> <0>" or utterance == "<0> <0> <0>":
            return utterance

        # Check if sentence ends on alphanumeric or % (e.g., "That adds up to 20%")
        if (utterance[-1].isalnum()) or (utterance[-1] == '%'):
            if lang == "zh":
                utterance += "。"
            else:
                utterance += "."

        return utterance

    #Adds <e> after designated end_character
    def add_terminator(self, utterance, end_characters):
        for end_character in end_characters:
            #Finds index of first end_character
            index = utterance.find(end_character)
            #Repeats while end_character remaining
            while index != -1:
                #Conditional satisfied if end_character is last in utterance
                if index+1 == len(utterance):
                    utterance = utterance + "<e>"
                # Conditional satisfied if not [char before is upper (end of abbreviation), char after is upper (start of abbreviation), or char after is numeric (indicating decimal) ]
                elif not (((index-1 >= 0) and utterance[index-1].isupper()) or ((index+1 < len(utterance)) and (utterance[index+1].isupper())) or ((index+1 < len(utterance)) and (utterance[index+1].isnumeric()))):
                    utterance = utterance[:index+1] + "<e>" + utterance[index+1:]
                index += 1
                index = utterance.find(end_character, index)

        return utterance

    #Returns the count of the number of sentences in an utterance
    def get_sentence_count(self, utterance):
        return utterance.count("<e>")


def process(args):
    root = Path(args.data_root).absolute()
    for lang in MUSTC.LANGUAGES:
        if lang not in args.langs_to_process:
            continue
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")
        audio_root.mkdir(exist_ok=True)

        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...", flush=True)
            dataset = MUSTC(root.as_posix(), lang, split, args.pair_type)
            if args.use_audio_input:
                print("Converting audios...", flush=True)
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    tgt_sample_rate = 16_000
                    _wavform, _ = convert_waveform(
                        waveform, sample_rate, to_mono=True,
                        to_sample_rate=tgt_sample_rate
                    )
                    sf.write(
                        (audio_root / f"{utt_id}.flac").as_posix(),
                        _wavform.T.numpy(), tgt_sample_rate
                    )
            else:
                print("Extracting log mel filter bank features...", flush=True)
                if split == 'train' and args.cmvn_type == "global":
                    gcmvn_feature_list = []
                    print("And estimating cepstral mean and variance stats...", flush=True)

                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    features = extract_fbank_features(
                        waveform, sample_rate, audio_root / f"{utt_id}.npy"
                    )
                    if split == 'train' and args.cmvn_type == "global":
                        if (len(gcmvn_feature_list) < args.gcmvn_max_num) and (features is not None):
                            gcmvn_feature_list.append(features)

                if split == 'train' and args.cmvn_type == "global":
                    # Estimate and save cmv
                    stats = cal_gcmvn_stats(gcmvn_feature_list)
                    with open(cur_root / "gcmvn.npz", "wb") as f:
                        np.savez(f, mean=stats["mean"], std=stats["std"])

        # Pack features into ZIP
        zip_path = cur_root / f"{audio_root.name}.zip"
        print("ZIPing audios/features...", flush=True)
        create_zip(audio_root, zip_path)
        print("Fetching ZIP manifest...", flush=True)
        audio_paths, audio_lengths = get_zip_manifest(
            zip_path,
            is_audio=args.use_audio_input,
        )
        # Generate TSV manifest
        print("Generating manifest...", flush=True)
        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split, args.pair_type)
            for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(audio_paths[utt_id])
                manifest["n_frames"].append(audio_lengths[utt_id])
                manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
                manifest["speaker"].append(speaker_id)
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            special_symbols = ['<0>', '<e>']
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
                special_symbols=special_symbols,
            )
        # Generate config YAML
        if args.use_audio_input:
            gen_config_yaml(
                cur_root,
                spm_filename=spm_filename_prefix + ".model",
                yaml_filename=f"config_{args.task}.yaml",
                specaugment_policy=None,
                extra={"use_audio_input": True}
            )
        else:
            gen_config_yaml(
                cur_root,
                spm_filename=spm_filename_prefix + ".model",
                yaml_filename=f"config_{args.task}.yaml",
                specaugment_policy="lb",
                cmvn_type=args.cmvn_type,
                gcmvn_path=(
                    cur_root / "gcmvn.npz" if args.cmvn_type == "global" else None
                ),
            )
        # Clean up
        shutil.rmtree(audio_root)


def process_joint(args):
    cur_root = Path(args.data_root)
    assert all((cur_root / f"en-{lang}").is_dir() for lang in MUSTC.LANGUAGES), \
        "do not have downloaded data available for all MUSTC languages"
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for lang in MUSTC.LANGUAGES:
            tsv_path = cur_root / f"en-{lang}" / f"train_{args.task}.tsv"
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        special_symbols = ['<0>', '<e>']
        if args.task == 'st':
            special_symbols += [f'<lang:{lang}>' for lang in MUSTC.LANGUAGES]
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols
        )
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        spm_filename=spm_filename_prefix + ".model",
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="ld",
        prepend_tgt_lang_tag=(args.task == "st"),
    )
    # Make symbolic links to manifests
    for lang in MUSTC.LANGUAGES:
        for split in MUSTC.SPLITS:
            src_path = cur_root / f"en-{lang}" / f"{split}_{args.task}.tsv"
            desc_path = cur_root / f"{split}_{lang}_{args.task}.tsv"
            if not desc_path.is_symlink():
                os.symlink(src_path, desc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=50000, type=int,
                        help=(
                            "Maximum number of sentences to use to estimate"
                            "global mean and variance"
                            ))
    parser.add_argument("--use-audio-input", action="store_true")
    parser.add_argument("--langs-to-process", nargs='+', default=[], help="List of MUSTC languages to process")
    parser.add_argument("--pair-type", default=None, type=str, help="Method to create paired sentence dataset, if desired")
    args = parser.parse_args()

    print(f"Args: {args}", flush=True)
    assert len(args.langs_to_process) > 0, "You must specify target language(s) using --langs-to-process"

    if args.joint:
        process_joint(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
