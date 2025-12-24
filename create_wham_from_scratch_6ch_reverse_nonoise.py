import os
import numpy as np
import soundfile as sf
import pandas as pd
from constants import SAMPLERATE
import argparse
from utils_nonoise import read_scaled_wav, quantize, fix_length, create_wham_mixes, append_or_truncate
from wham_room import WhamRoom


FILELIST_STUB = os.path.join('data', 'mix_2_spk_filenames_{}.csv')

# SINGLE_DIR = 'mix_single'  # Removed: contains noise
# BOTH_DIR = 'mix_both'  # Removed: contains noise
CLEAN_DIR = 'mix_clean'
# BOTH_REVERSE_DIR = 'mix_both_reverse'  # Removed: contains noise
CLEAN_REVERSE_DIR = 'mix_clean_reverse'
S1_DIR = 's1'
S2_DIR = 's2'
# NOISE_DIR = 'noise'  # Removed: no noise
SUFFIXES = ['_reverb']

MONO = False  # Generate mono audio, change to false for stereo audio
SPLITS = ['tr', 'cv', 'tt']
SAMPLE_RATES = ['8k'] # Remove element from this list to generate less data
DATA_LEN = ['min'] # Remove element from this list to generate less data

def create_wham(wsj_root, output_root):
    LEFT_CH_IND = 0
    if MONO:
        ch_ind = LEFT_CH_IND
    else:
        ch_ind = [0, 1, 2, 3, 4, 5]  # 6 channels

    # Note: scaling_npz is not needed for no-noise version, but we still need utterance_id and start_samp_16k
    # For now, we'll use a dummy approach or read from a minimal metadata file
    # If scaling_npz is still needed for utterance_id, we need to handle it differently
    reverb_param_stub = os.path.join('data', 'reverb_params_{}_6ch.csv')

    for splt in SPLITS:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        reverb_param_path = reverb_param_stub.format(splt)
        reverb_param_df = pd.read_csv(reverb_param_path)

        for wav_dir in ['wav' + sr for sr in SAMPLE_RATES]:
            for datalen_dir in DATA_LEN:
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                for sfx in SUFFIXES:
                    os.makedirs(os.path.join(output_path, CLEAN_DIR+sfx), exist_ok=True)
                    # os.makedirs(os.path.join(output_path, SINGLE_DIR+sfx), exist_ok=True)  # Removed: contains noise
                    # os.makedirs(os.path.join(output_path, BOTH_DIR+sfx), exist_ok=True)  # Removed: contains noise
                    os.makedirs(os.path.join(output_path, CLEAN_REVERSE_DIR+sfx), exist_ok=True)
                    # os.makedirs(os.path.join(output_path, BOTH_REVERSE_DIR+sfx), exist_ok=True)  # Removed: contains noise
                    os.makedirs(os.path.join(output_path, S1_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, S2_DIR+sfx), exist_ok=True)
                # os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)  # Removed: no noise

        # Use utterance_id from wsjmix_df instead of scaling_npz
        utt_ids = wsjmix_df['output_filename'].values
        # For no-noise version, we don't need start_samp_16k from scaling_npz
        # We'll use 0 as default or calculate from speech files
        start_samp_16k = np.zeros(len(utt_ids), dtype=int)

        for i_utt, output_name in enumerate(utt_ids):
            utt_row = reverb_param_df[reverb_param_df['utterance_id'] == output_name]
            mic_z = utt_row['mic_z'].iloc[0]
            mics = [
                [utt_row['micL_x'].iloc[0], utt_row['micL_y'].iloc[0], mic_z],
                [utt_row['mic3_x'].iloc[0], utt_row['mic3_y'].iloc[0], mic_z],
                [utt_row['mic4_x'].iloc[0], utt_row['mic4_y'].iloc[0], mic_z],
                [utt_row['micR_x'].iloc[0], utt_row['micR_y'].iloc[0], mic_z],
                [utt_row['mic5_x'].iloc[0], utt_row['mic5_y'].iloc[0], mic_z],
                [utt_row['mic6_x'].iloc[0], utt_row['mic6_y'].iloc[0], mic_z],
            ]
            # Create normal room: s1 at s1 position, s2 at s2 position
            room = WhamRoom([utt_row['room_x'].iloc[0], utt_row['room_y'].iloc[0], utt_row['room_z'].iloc[0]],
                            mics,
                            [utt_row['s1_x'].iloc[0], utt_row['s1_y'].iloc[0], utt_row['s1_z'].iloc[0]],
                            [utt_row['s2_x'].iloc[0], utt_row['s2_y'].iloc[0], utt_row['s2_z'].iloc[0]],
                            utt_row['T60'].iloc[0])
            room.generate_rirs()
            
            # Create reverse room: s2 position for source 0 (s1 audio), s1 position for source 1 (s2 audio)
            room_reverse = WhamRoom([utt_row['room_x'].iloc[0], utt_row['room_y'].iloc[0], utt_row['room_z'].iloc[0]],
                                    mics,
                                    [utt_row['s2_x'].iloc[0], utt_row['s2_y'].iloc[0], utt_row['s2_z'].iloc[0]],
                                    [utt_row['s1_x'].iloc[0], utt_row['s1_y'].iloc[0], utt_row['s1_z'].iloc[0]],
                                    utt_row['T60'].iloc[0])
            room_reverse.generate_rirs()

            # read the 16kHz unscaled speech files, but make sure to add all 'max' padding to end of utterances
            # for synthesizing all the reverb tails
            utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
            s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
            s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])
            s1_temp = quantize(read_scaled_wav(s1_path, 1))
            s2_temp = quantize(read_scaled_wav(s2_path, 1))
            s1_temp, s2_temp = fix_length(s1_temp, s2_temp, 'max')
            # No noise processing needed

            # Add audio to both rooms
            room.add_audio(s1_temp, s2_temp)
            room_reverse.add_audio(s1_temp, s2_temp)

            # Generate audio from both rooms
            anechoic = room.generate_audio(anechoic=True, fs=SAMPLE_RATES)
            reverberant = room.generate_audio(fs=SAMPLE_RATES)
            anechoic_reverse = room_reverse.generate_audio(anechoic=True, fs=SAMPLE_RATES)
            reverberant_reverse = room_reverse.generate_audio(fs=SAMPLE_RATES)

            for sr_i, sr_dir in enumerate(SAMPLE_RATES):
                wav_dir = 'wav' + sr_dir
                if sr_dir == '8k':
                    sr = 8000
                    downsample = True
                else:
                    sr = SAMPLERATE
                    downsample = False

                for datalen_dir in DATA_LEN:
                    output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)

                    # For no-noise version, we use simple scaling (1.0) or can be adjusted
                    # Original scaling factors are not needed without noise
                    utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
                    s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
                    s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])

                    s1 = read_scaled_wav(s1_path, 1.0, downsample)
                    s1 = quantize(s1)
                    s2 = read_scaled_wav(s2_path, 1.0, downsample)
                    s2 = quantize(s2)

                    # Make relative source energy of anechoic sources same with original in mono (left channel) case
                    # Note: positions are reversed in reverse room
                    # Source 0 (s1 position) has s1 audio, so s1 is at index 0 in normal room
                    # Source 1 (s2 position) has s2 audio, so s2 is at index 1 in normal room
                    s1_spatial_scaling = np.sqrt(np.sum(s1 ** 2) / np.sum(anechoic[sr_i][0, LEFT_CH_IND, :] ** 2))
                    s2_spatial_scaling = np.sqrt(np.sum(s2 ** 2) / np.sum(anechoic[sr_i][1, LEFT_CH_IND, :] ** 2))
                    
                    # For reverse room: Source 0 (s2 position) has s1 audio, Source 1 (s1 position) has s2 audio
                    s1_spatial_scaling_reverse = np.sqrt(np.sum(s1 ** 2) / np.sum(anechoic_reverse[sr_i][0, LEFT_CH_IND, :] ** 2))
                    s2_spatial_scaling_reverse = np.sqrt(np.sum(s2 ** 2) / np.sum(anechoic_reverse[sr_i][1, LEFT_CH_IND, :] ** 2))

                    # No noise processing
                    if datalen_dir == 'max':
                        out_len = np.maximum(len(s1), len(s2))
                    else:
                        out_len = np.minimum(len(s1), len(s2))

                    # Process normal room audio
                    s1_anechoic, s2_anechoic = fix_length(anechoic[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                          anechoic[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                          datalen_dir)
                    s1_reverb, s2_reverb = fix_length(reverberant[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                      reverberant[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                      datalen_dir)
                    
                    # Process reverse room audio
                    # Note: In reverse room, Source 0 (s2 position) has s1 audio, Source 1 (s1 position) has s2 audio
                    s1_anechoic_reverse, s2_anechoic_reverse = fix_length(anechoic_reverse[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling_reverse,
                                                                          anechoic_reverse[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling_reverse,
                                                                          datalen_dir)
                    s1_reverb_reverse, s2_reverb_reverse = fix_length(reverberant_reverse[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling_reverse,
                                                                      reverberant_reverse[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling_reverse,
                                                                      datalen_dir)

                    # Process normal room sources
                    sources = [(s1_anechoic, s2_anechoic), (s1_reverb, s2_reverb)]
                    sources_reverse = [(s1_anechoic_reverse, s2_anechoic_reverse), (s1_reverb_reverse, s2_reverb_reverse)]
                    
                    for i_sfx, (sfx, source_pair, source_pair_reverse) in enumerate(zip(SUFFIXES, sources, sources_reverse)):
                        # Process normal room
                        s1_samples, s2_samples = append_or_truncate(source_pair[0], source_pair[1],
                                                                   datalen_dir,
                                                                   start_samp_16k[i_utt], downsample)

                        mix_clean = create_wham_mixes(s1_samples, s2_samples)

                        # Write normal room audio
                        samps = [mix_clean, s1_samples, s2_samples]
                        dirs = [CLEAN_DIR, S1_DIR, S2_DIR]
                        for dir, samp in zip(dirs, samps):
                            sf.write(os.path.join(output_path, dir+sfx, output_name), samp,
                                     sr, subtype='FLOAT')

                        # Process reverse room
                        s1_samples_reverse, s2_samples_reverse = append_or_truncate(source_pair_reverse[0], source_pair_reverse[1],
                                                                                    datalen_dir,
                                                                                    start_samp_16k[i_utt], downsample)

                        mix_clean_reverse = create_wham_mixes(s1_samples_reverse, s2_samples_reverse)

                        # Write reverse room audio (only mix_clean_reverse)
                        sf.write(os.path.join(output_path, CLEAN_REVERSE_DIR+sfx, output_name), mix_clean_reverse,
                                 sr, subtype='FLOAT')

            if (i_utt + 1) % 500 == 0:
                print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--wsj0-root', type=str, required=True,
                        help='Path to the folder containing wsj0/')
    # --wham-noise-root removed: no noise needed
    args = parser.parse_args()
    create_wham(args.wsj0_root, args.output_dir)
