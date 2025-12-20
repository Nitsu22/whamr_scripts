import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path('data/reverb_params_cv_6ch.csv')
DEFAULT_OUTPUT = Path('data/reverb_params_cv_6ch_spkpos.csv')
DEFAULT_SEED = 17


def randomize_positions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomize s1/s2 positions while keeping them inside the room bounds."""
    df = df.copy()
    room_x = df['room_x'].to_numpy(float)
    room_y = df['room_y'].to_numpy(float)
    room_z = df['room_z'].to_numpy(float)

    df['s1_x'] = rng.random(len(df)) * room_x
    df['s1_y'] = rng.random(len(df)) * room_y
    df['s1_z'] = rng.random(len(df)) * room_z

    df['s2_x'] = rng.random(len(df)) * room_x
    df['s2_y'] = rng.random(len(df)) * room_y
    df['s2_z'] = rng.random(len(df)) * room_z

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Randomize s1/s2 positions within the room for a 6-channel reverb parameter CSV.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Input CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output CSV (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.input)
    randomized_df = randomize_positions(df, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    randomized_df.to_csv(args.output, index=False)
    print(f'Wrote {len(randomized_df)} rows to {args.output}')


if __name__ == '__main__':
    main()
