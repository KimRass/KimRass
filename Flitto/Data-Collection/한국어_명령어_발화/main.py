import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from utils.audio_utils import (
    read_audio,
    normalize_signal,
    _signal_to_df,
    get_spectrogram,
    get_melspectrogram,
    _spectrogram_to_df,
    _melspectrogram_to_df,
    get_utterance_periods_and_silence_periods,
    get_plosive_periods,
    get_period_of_unrecorded_frequencies,
    draw_multiple_features
)
from speech_ko_command_utterances.utils import get_args, get_command_and_speed
from logger import Logger


def save_multiple_sheets(df_wav, df_spec, df_mel_db, dir, audio_path):
    xlsx_path = dir / "values" / audio_path.parent.name / f"{audio_path.stem}.xlsx"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
    df_wav.to_excel(writer, sheet_name="waveform")
    df_spec.to_excel(writer, sheet_name="spectrogram")
    df_mel_db.to_excel(writer, sheet_name="melspectrogram")
    writer.save()


def save_features(dir):
    dir = Path(dir)

    processing_dir = dir / "processing"
    ls_audio_path = sorted(dir.glob("processing/**/*.wav"))
    features_dir = dir / "features"
    ls_feature_path = sorted(features_dir.glob("*/*.png"))
    if len(ls_feature_path) == len(ls_audio_path):
        logger.info(f"Total {len(ls_feature_path):,} feature(s) already exist")
    else:
        ls_subdir = [
            dir for dir in sorted(list(processing_dir.iterdir())) if dir.is_dir()
        ]
        for i, subdir in enumerate(ls_subdir, start=1):
            if subdir.is_dir():
                logger.info(f"[{i}/{len(ls_subdir)}] Processing '{subdir.name}'...")
                for audio_path in tqdm(sorted(subdir.glob("*.wav"))):
                    png_path = (
                        features_dir / subdir.name / f"{audio_path.stem}.png"
                    )
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    if png_path.exists():
                        continue
                    else:
                        _, speed = get_command_and_speed(audio_path)

                        y, sr = read_audio(audio_path)
                        y_norm = normalize_signal(y)
                        df_wav = _signal_to_df(signal=y, sr=sr)
                        spec_db = get_spectrogram(
                            signal_norm=y_norm, sr=sr, unit="db"
                        )
                        df_spec_db = _spectrogram_to_df(
                            spec=spec_db, sr=sr
                        )
                        mel_db = get_melspectrogram(signal_norm=y_norm, sr=sr, unit="db")
                        df_mel_db = _melspectrogram_to_df(
                            mel=mel_db, sr=sr
                        )
                        period_unrecorded_freqs = get_period_of_unrecorded_frequencies(
                            df_spec_db
                        )

                        utterance_periods, silence_periods = get_utterance_periods_and_silence_periods(
                            signal=y,
                            sr=sr,
                            df_mel_db=df_mel_db,
                            adjacent_if_lower_than=0.25 if speed == "fast" else 0.35
                        )
                        plosive_periods = get_plosive_periods(
                            df_spec_db=df_spec_db,
                            sr=sr,
                        )

                        draw_multiple_features(
                            df_wav=df_wav,
                            spec_db=spec_db,
                            mel_db=mel_db,
                            utterance_periods=utterance_periods,
                            silence_periods=silence_periods,
                            plosive_periods=plosive_periods,
                            period_unrecorded_freqs=period_unrecorded_freqs,
                            sr=sr,
                        )
                        plt.savefig(png_path, bbox_inches="tight")
                        plt.close()
        logger.info("Completed saving features!")


def main():
    args = get_args()

    global logger
    logger = Logger(args.dir).get_logger()

    save_features(args.dir)


if __name__ == "__main__":
    main()