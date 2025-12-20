import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASE = Path('data/reverb_params_cv_6ch.csv')
DEFAULT_RANDOM = Path('data/reverb_params_cv_6ch_spkpos.csv')
DEFAULT_UTT = '01to030v_0.76421_20ga010m_-0.76421.wav'
DEFAULT_OUT = Path('data/plots/utterance_positions.png')


def read_row(csv_path: Path, utt_id: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    row = df[df['utterance_id'] == utt_id]
    if row.empty:
        raise ValueError(f'utterance_id {utt_id} not found in {csv_path}')
    return row.iloc[0]


def get_positions(row: pd.Series) -> Tuple[float, float, dict]:
    room_x = float(row['room_x'])
    room_y = float(row['room_y'])
    positions = {
        'mics': {
            'micL': (row['micL_x'], row['micL_y']),
            'mic3': (row['mic3_x'], row['mic3_y']),
            'mic4': (row['mic4_x'], row['mic4_y']),
            'micR': (row['micR_x'], row['micR_y']),
            'mic5': (row['mic5_x'], row['mic5_y']),
            'mic6': (row['mic6_x'], row['mic6_y']),
        },
        's1': (row['s1_x'], row['s1_y']),
        's2': (row['s2_x'], row['s2_y']),
    }
    return room_x, room_y, positions


def plot_scene(ax, room_x: float, room_y: float, positions: dict, title: str):
    ax.set_title(title)
    # Room outline
    ax.plot([0, room_x, room_x, 0, 0], [0, 0, room_y, room_y, 0], color='black', linewidth=1)

    mic_names = ['micL', 'mic3', 'mic4', 'micR', 'mic5', 'mic6']
    mic_x = [positions['mics'][name][0] for name in mic_names]
    mic_y = [positions['mics'][name][1] for name in mic_names]
    ax.scatter(mic_x, mic_y, c='tab:blue', marker='o', label='Mics')
    for name, x, y in zip(mic_names, mic_x, mic_y):
        ax.text(x, y, name, fontsize=8, ha='center', va='bottom')

    ax.scatter([positions['s1'][0]], [positions['s1'][1]], c='tab:orange', marker='^', label='s1')
    ax.scatter([positions['s2'][0]], [positions['s2'][1]], c='tab:green', marker='s', label='s2')

    ax.set_xlim(-room_x * 0.05, room_x * 1.05)
    ax.set_ylim(-room_y * 0.05, room_y * 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description='Plot mic and speaker positions for a specific utterance from two CSVs (base vs randomized).'
    )
    parser.add_argument('--base-csv', type=Path, default=DEFAULT_BASE,
                        help=f'Original CSV path (default: {DEFAULT_BASE})')
    parser.add_argument('--random-csv', type=Path, default=DEFAULT_RANDOM,
                        help=f'Randomized CSV path (default: {DEFAULT_RANDOM})')
    parser.add_argument('--utterance-id', type=str, default=DEFAULT_UTT,
                        help=f'Utterance ID to plot (default: {DEFAULT_UTT})')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT,
                        help=f'Output image path (default: {DEFAULT_OUT})')
    args = parser.parse_args()

    base_row = read_row(args.base_csv, args.utterance_id)
    rand_row = read_row(args.random_csv, args.utterance_id)

    base_room_x, base_room_y, base_pos = get_positions(base_row)
    rand_room_x, rand_room_y, rand_pos = get_positions(rand_row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_scene(axes[0], base_room_x, base_room_y, base_pos, 'Original')
    plot_scene(axes[1], rand_room_x, rand_room_y, rand_pos, 'Randomized')
    fig.suptitle(f'Positions for {args.utterance_id}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f'Saved plot to {args.output}')


if __name__ == '__main__':
    main()
