import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# python generate_reverb_params_6ch.py

DEFAULT_INPUT = Path('data/reverb_params_tt.csv')
DEFAULT_OUTPUT = Path('data/reverb_params_tt_6ch.csv')


def add_surround_mics(df: pd.DataFrame) -> pd.DataFrame:
    """Add mic3-6 coordinates on the circle with micL-micR as diameter."""
    mic_l = df[['micL_x', 'micL_y']].to_numpy(float)
    mic_r = df[['micR_x', 'micR_y']].to_numpy(float)
    center = (mic_l + mic_r) / 2.0
    radius = np.linalg.norm(mic_l - mic_r, axis=1) / 2.0

    theta0 = np.arctan2(mic_l[:, 1] - center[:, 1], mic_l[:, 0] - center[:, 0])
    step = 2 * np.pi / 6.0
    angles = theta0[:, None] - step * np.arange(6)[None, :]

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    offsets = np.stack([cos_vals, sin_vals], axis=2) * radius[:, None, None]
    positions = center[:, None, :] + offsets  # shape: (n_rows, 6, 2)

    df = df.copy()
    df['mic3_x'] = positions[:, 1, 0]
    df['mic3_y'] = positions[:, 1, 1]
    df['mic4_x'] = positions[:, 2, 0]
    df['mic4_y'] = positions[:, 2, 1]
    df['mic5_x'] = positions[:, 4, 0]
    df['mic5_y'] = positions[:, 4, 1]
    df['mic6_x'] = positions[:, 5, 0]
    df['mic6_y'] = positions[:, 5, 1]

    column_order = [
        'utterance_id',
        'room_x',
        'room_y',
        'room_z',
        'micL_x',
        'micL_y',
        'mic3_x',
        'mic3_y',
        'mic4_x',
        'mic4_y',
        'micR_x',
        'micR_y',
        'mic5_x',
        'mic5_y',
        'mic6_x',
        'mic6_y',
        'mic_z',
        's1_x',
        's1_y',
        's1_z',
        's2_x',
        's2_y',
        's2_z',
        'T60',
    ]

    return df[column_order]


def main():
    parser = argparse.ArgumentParser(
        description='Create a 6-channel reverb parameter file by placing mics on the circle defined by micL-micR.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help=f'Input reverb parameter CSV (default: {DEFAULT_INPUT})',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output CSV path (default: {DEFAULT_OUTPUT})',
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    df = pd.read_csv(input_path)
    output_df = add_surround_mics(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f'Wrote {len(output_df)} rows to {output_path}')


if __name__ == '__main__':
    main()
