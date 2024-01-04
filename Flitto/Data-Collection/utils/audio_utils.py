import math
import soundfile as sf
import copy
import wavio
import noisereduce as nr
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

global FRAME_LEN, FRAME_SHIFT
FRAME_LEN = 20
FRAME_SHIFT = 10
N_MELS = 80

global MIN_AMP, MAX_AMP
MIN_AMP = 3000
MAX_AMP = 20000


def read_audio(audio_path):
    ext = Path(audio_path).suffix
    if ext in [".wav", ".flac"]:
        y, sr = sf.read(audio_path, dtype="int16")
        if y.ndim == 2:
            print("This audio file is stereo!")
        y = y.astype(np.int32)
    elif ext == ".pcm":
        y = np.memmap(audio_path, dtype="h", mode="r")
    return y, sr


def write_audio(save_path, signal, sr) -> None:
    wavio.write(
        file=str(save_path),
        data=signal.astype(np.int16),
        rate=sr,
        scale="none"
    )


def _signal_to_df(signal, sr):
    df_wav = pd.DataFrame(signal, columns=["signal"])
    df_wav["time"] = df_wav.index / sr
    return df_wav[["time", "signal"]]


def normalize_signal(signal):
    return signal / (2 ** 15)


def _spectrogram_to_df(spec, sr):
    n_fft = int(round(sr * 0.001 * FRAME_LEN))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).round().astype("int")
    return pd.DataFrame(spec, index=freqs)


def _melspectrogram_to_df(mel, sr):
    freqs = librosa.mel_frequencies(N_MELS=N_MELS, fmax=sr // 2).round().astype("int")
    return pd.DataFrame(mel, index=freqs)


def get_spectrogram(signal_norm, sr, unit="db"):
    n_fft = int(round(sr * 0.001 * FRAME_LEN))
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    stft = librosa.stft(signal_norm, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(stft)
    if unit == "linear":
        return spec
    elif unit == "db":
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        return spec_db
    else:
        return np.empty(spec.shape)


def get_number_of_frames_of_feature(n_waveform_frames, sr):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))
    return int(math.ceil(n_waveform_frames / hop_length))


def _waveform_frame_scale_to_feature_frame_scale(x, sr):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))
    return np.ceil(x / hop_length).astype("int")


def _feature_frame_scale_to_waveform_frame_scale(x, sr):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))
    return x * hop_length


def _waveform_frame_scale_to_time_scale(x, sr):
    return x / sr


def _time_scale_to_waveform_frame_scale(x, sr):
    return x * sr


def _feature_frame_scale_to_time_scale(x, sr):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))
    return x * hop_length / sr


def _time_scale_to_feature_frame_scale(x, sr):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))
    return x * sr / hop_length


def get_melspectrogram(signal_norm, sr, unit="db"):
    n_fft = int(round(sr * 0.001 * FRAME_LEN))
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    mel = librosa.feature.melspectrogram(
        y=signal_norm, sr=sr, n_fft=n_fft, hop_length=hop_length, N_MELS=N_MELS
    )
    if unit == "linear":
        return mel
    elif unit == "db":
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        return mel_db
    else:
        n_waveform_frames = len(signal_norm)
        n_feature_frames = get_number_of_frames_of_feature(
            n_waveform_frames=n_waveform_frames, sr=sr
        )
        return np.empty((N_MELS, n_feature_frames))


def draw_horizontal_line(ax, y):
    ax.axhline(y, color="red", ls="--", lw=0.5)


def draw_vertical_line(ax, x):
    ax.axvline(x, color="green", ls="--", lw=0.8)


def draw_absolute_waveform(
    df_wav,
    sr,
    utterance_periods=pd.DataFrame(columns=["from", "to"]),
    silence_periods=pd.DataFrame(columns=["from", "to"]),
    MIN_AMP=MIN_AMP,
    MAX_AMP=MAX_AMP,
    ax=plt.gca(),
    duration=5,
):
    df_wav = df_wav.set_index("time")
    ax.axvspan(xmin=0, xmax=df_wav.index[-1], alpha=1, color="black")
    df_wav["signal"].abs().plot.line(ax=ax)
    for v in [MIN_AMP, MAX_AMP]:
        draw_horizontal_line(ax=ax, y=v)
    utterance_periods = _waveform_frame_scale_to_time_scale(
        x=utterance_periods, sr=sr
    )
    silence_periods = _waveform_frame_scale_to_time_scale(
        x=silence_periods, sr=sr
    )
    for _, row in utterance_periods.iterrows():
        draw_vertical_line(ax=ax, x=row["from"])
        draw_vertical_line(ax=ax, x=row["to"])
        ax.axvspan(xmin=row["from"], xmax=row["to"], alpha=0.15, color="green")
    if not utterance_periods.empty:
        ax.text(
            x=(utterance_periods.iloc[0, 0] + utterance_periods.iloc[-1, -1]) / 2,
            ha="center",
            y=30000,
            s=round(utterance_periods.iloc[-1, -1] - utterance_periods.iloc[0, 0], 2),
            fontsize=14,
            color="white"
        )
    if not silence_periods.empty:
        ax.text(
            x=(silence_periods.iloc[0, 0] + silence_periods.iloc[0, 1]) / 2,
            ha="center",
            y=30000,
            s=round(silence_periods.iloc[0, 1] - silence_periods.iloc[0, 0], 2),
            fontsize=14,
            color="white"
        )
        ax.text(
            x=(silence_periods.iloc[-1, 0] + silence_periods.iloc[-1, 1]) / 2,
            ha="center",
            y=30000,
            s=round(silence_periods.iloc[-1, 1] - silence_periods.iloc[-1, 0], 2),
            fontsize=14,
            color="white"
        )

    if ax == plt.gca():
        plt.xlim([0, duration])
        plt.ylim([0, 2 ** 15])
        plt.xticks([round(i * 0.1, 1) for i in range(int(duration / 0.1))], rotation=90)
    else:
        ax.set(xlim=[0, duration], ylim=[0, 2 ** 15])
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter("{x:.1f}")
        ax.tick_params(axis="x", which="major", length=10, width=2, labelrotation=90)

        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", which="minor", length=7, width=1)


def draw_spectrogram(
    spec,
    sr,
    utterance_periods=pd.DataFrame(columns=["from", "to"]),
    plosive_periods=pd.DataFrame(columns=["from", "to"]),
    unrecorded_frequencies=pd.DataFrame(columns=["from", "to"]),
    scale="log",
    duration=5,
    shows_colorbar=True,
    ax=plt.gca(),
    frequencies_with_noise=pd.DataFrame(columns=["from", "to"]),
):
    n_fft = int(round(sr * 0.001 * FRAME_LEN))
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    n_waveform_frames = duration * sr
    n_feature_frames = get_number_of_frames_of_feature(
        n_waveform_frames=n_waveform_frames, sr=sr
    )

    img = librosa.display.specshow(
        spec,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="frames",
        y_axis=scale,
        cmap="inferno",
        ax=ax,
    )

    utterance_periods = _waveform_frame_scale_to_feature_frame_scale(
        x=utterance_periods, sr=sr
    )
    for _, row in utterance_periods.iterrows():
        draw_vertical_line(ax=ax, x=row["from"])
        draw_vertical_line(ax=ax, x=row["to"])
        ax.axvspan(row["from"], row["to"], alpha=0.15, color="green")
    for _, row in unrecorded_frequencies.iterrows():
        ax.axhspan(
            ymin=row["from"], ymax=row["to"], xmin=0, xmax=0.01, alpha=1, color="red"
        )
    for _, row in frequencies_with_noise.iterrows():
        ax.axhspan(
            ymin=row["from"], ymax=row["to"], xmin=0, xmax=0.01, alpha=1, color="blue"
        )
    for _, row in plosive_periods.iterrows():
        ax.axvspan(row["from"], row["to"], alpha=0.15, color="yellow")

    if ax == plt.gca():
        plt.xlim([0, n_feature_frames])
    else:
        ax.set(xlim=[0, n_feature_frames])
    if shows_colorbar:
        plt.colorbar(img, format="%+2.0f dB", ax=ax)


def draw_melspectrogram(
    mel,
    sr,
    utterance_periods=pd.DataFrame(columns=["from", "to"]),
    plosive_periods=pd.DataFrame(columns=["from", "to"]),
    scale="mel",
    shows_colorbar=True,
    ax=plt.gca(),
    duration=3.5,
):
    n_fft = int(round(sr * 0.001 * FRAME_LEN))
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    n_waveform_frames = duration * sr
    n_feature_frames = get_number_of_frames_of_feature(
        n_waveform_frames=n_waveform_frames, sr=sr
    )

    img = librosa.display.specshow(
        mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="frames",
        y_axis=scale,
        cmap="inferno",
        ax=ax,
    )

    utterance_periods = _waveform_frame_scale_to_feature_frame_scale(
        utterance_periods, sr=sr
    )
    for _, row in utterance_periods.iterrows():
        draw_vertical_line(ax=ax, x=row["from"])
        draw_vertical_line(ax=ax, x=row["to"])
        ax.axvspan(row["from"], row["to"], alpha=0.15, color="green")
    for _, row in plosive_periods.iterrows():
        ax.axvspan(row["from"], row["to"], alpha=0.15, color="yellow")

    if ax == plt.gca():
        plt.xlim([0, n_feature_frames])
    else:
        ax.set(xlim=[0, n_feature_frames])
    if shows_colorbar:
        plt.colorbar(img, format="%+2.0f dB", ax=ax)


def clip_greater_than_or_equal_to(df_db, threshold=-60):
    return df_db.where(df_db < threshold, threshold)


def get_frequencies_with_noise(df_spec_db):
    df_spec_db = clip_greater_than_or_equal_to(df_spec_db)
    df_spec_db = df_spec_db + 80
    df_spec_db["mean"] = df_spec_db.mean(axis=1)
    df_spec_db["std"] = df_spec_db.std(axis=1)
    df_spec_db["temp"] = df_spec_db.apply(
        lambda x: x["mean"] / (x["std"] ** 2) if x["std"] != 0 else 0, axis=1
    )

    ls_freq = df_spec_db[
        (df_spec_db["mean"] >= 6) & (df_spec_db["temp"] >= 0.2)
    ].index.tolist()
    ls_freq_from_to = list()
    for frame in ls_freq:
        ls_freq_from_to = add_single_value_to_intervals(
            list_of_intervals=ls_freq_from_to,
            value=frame,
            allows_less_than_or_equal_to=50,
        )
    return pd.DataFrame(ls_freq_from_to, columns=["from", "to"])


def draw_multiple_features(
    df_wav,
    spec_db,
    mel_db,
    utterance_periods,
    silence_periods,
    plosive_periods,
    period_unrecorded_freqs,
    sr,
    duration=3.5
):
    _, axes = plt.subplots(3, 1, figsize=(18, 16))
    draw_absolute_waveform(
        df_wav=df_wav,
        sr=sr,
        duration=duration,
        utterance_periods=utterance_periods,
        silence_periods=silence_periods,
        ax=axes[0],
    )
    df_spec_db = _spectrogram_to_df(
        spec=spec_db, sr=sr
    )
    frequencies_with_noise = get_frequencies_with_noise(df_spec_db)
    draw_spectrogram(
        spec=spec_db,
        sr=sr,
        utterance_periods=utterance_periods,
        plosive_periods=plosive_periods,
        unrecorded_frequencies=period_unrecorded_freqs,
        scale="linear",
        duration=duration,
        shows_colorbar=False,
        ax=axes[1],
        frequencies_with_noise=frequencies_with_noise,
    )
    draw_melspectrogram(
        mel=mel_db,
        sr=sr,
        utterance_periods=utterance_periods,
        plosive_periods=plosive_periods,
        scale="mel",
        duration=duration,
        shows_colorbar=False,
        ax=axes[2],
    )


def integrate_adjacent_periods_in_waveform_frame_scale(df, sr, adjacent_if_lower_than):
    adjacent_if_lower_than = _time_scale_to_waveform_frame_scale(
        adjacent_if_lower_than, sr
    )
    if not df.empty:
        while True:
            leng_bef = len(df)

            df["from_shift"] = df["from"].shift(-1)
            df["to_shift"] = df["to"].shift(-1)

            df["to"] = df.apply(
                lambda x: x["to_shift"]
                if x["from_shift"] - x["to"] < adjacent_if_lower_than
                else x["to"],
                axis=1,
            )
            df.drop_duplicates(["to"], keep="first", inplace=True)

            leng_aft = len(df)
            if leng_bef == leng_aft:
                break
            leng_bef = leng_aft
    return df[["from", "to"]].astype("int")


def get_utterance_periods_in_waveform_frame_scale_using_waveform(
    signal_norm, sr, adjacent_if_lower_than=0.3, top_db=38
):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    utterance_periods = pd.DataFrame(
        librosa.effects.split(
            signal_norm, top_db=top_db, FRAME_LEN=FRAME_LEN, hop_length=hop_length
        ),
        columns=["from", "to"],
    )
    if not utterance_periods.empty:
        utterance_periods["duration"] = utterance_periods.apply(
            lambda x: _waveform_frame_scale_to_time_scale(
                x["to"] - x["from"], sr
            ),
            axis=1,
        )
        utterance_periods = utterance_periods[utterance_periods["duration"] >= 0.1]
    utterance_periods = integrate_adjacent_periods_in_waveform_frame_scale(
        utterance_periods, sr, adjacent_if_lower_than
    )
    return utterance_periods[["from", "to"]]


def delete_lower_than(df_db, threshold=-40):
    return df_db.where(df_db >= threshold, -80)


def get_utterance_periods_in_waveform_frame_scale_using_melspectrogram(
    df_mel_db, sr, adjacent_if_lower_than=0.3
):
    adjacent_if_lower_than = _time_scale_to_feature_frame_scale(
        adjacent_if_lower_than, sr
    )

    df_mel_db = delete_lower_than(df_mel_db)
    sum_line = df_mel_db.sum(axis=0)
    threshold = sum_line.median() + 100
    ls_feature_frames = sum_line[sum_line > threshold].index.tolist()
    ls_utterance_periods = list()
    for frame in ls_feature_frames:
        ls_utterance_periods = add_single_value_to_intervals(
            list_of_intervals=ls_utterance_periods,
            value=frame,
            allows_less_than_or_equal_to=adjacent_if_lower_than,
        )
    utterance_periods = pd.DataFrame(ls_utterance_periods, columns=["from", "to"])
    utterance_periods = _feature_frame_scale_to_waveform_frame_scale(
        utterance_periods, sr=sr
    )

    if not utterance_periods.empty:
        utterance_periods["duration"] = utterance_periods.apply(
            lambda x: _waveform_frame_scale_to_time_scale(
                x["to"] - x["from"], sr
            ),
            axis=1,
        )
        utterance_periods = utterance_periods[utterance_periods["duration"] >= 0.1]
    return utterance_periods[["from", "to"]]


def get_stable_utterance_periods_in_waveform_frame_scale(df_mel, sr, utterance_periods):
    utterance_periods = _waveform_frame_scale_to_feature_frame_scale(
        utterance_periods, sr=sr
    )

    cos_sim_line = get_cosine_similarities_along_column(df_mel)
    ser_cos_sim_line = pd.Series(cos_sim_line)
    ser_diff = ser_cos_sim_line.diff(-1).abs()

    ls_stable_utterance_periods = list()
    for i in ser_diff[ser_diff < 0.005].index:
        for _, row in utterance_periods.iterrows():
            if row["from"] <= i and i < row["to"]:
                ls_stable_utterance_periods = add_single_value_to_intervals(
                    list_of_intervals=ls_stable_utterance_periods,
                    value=i,
                    allows_less_than_or_equal_to=2,
                )
    stable_utterance_periods = pd.DataFrame(
        ls_stable_utterance_periods, columns=["from", "to"]
    )
    stable_utterance_periods = _feature_frame_scale_to_waveform_frame_scale(
        stable_utterance_periods, sr=sr
    )
    return stable_utterance_periods


def get_silence_periods_in_waveform_frame_scale(utterance_periods, signal):
    utterance_periods = pd.concat(
        [pd.DataFrame({"from": 0, "to": 0}, index=[0]), utterance_periods],
        ignore_index=True,
    )
    leng_signal = len(signal)
    utterance_periods = pd.concat(
        [utterance_periods, pd.DataFrame({"from": leng_signal, "to": leng_signal}, index=[0])],
        ignore_index=True,
    )
    utterance_periods["from_shift"] = utterance_periods["from"].shift(-1)
    utterance_periods["from"] = utterance_periods["to"]
    utterance_periods["to"] = utterance_periods["from_shift"]
    utterance_periods = utterance_periods.iloc[:-1]
    return utterance_periods[["from", "to"]].astype("int")


def get_utterance_periods_and_silence_periods(signal, sr, df_mel_db, adjacent_if_lower_than=0.3):
    utterance_periods = get_utterance_periods_in_waveform_frame_scale_using_melspectrogram(
        df_mel_db=df_mel_db, sr=sr, adjacent_if_lower_than=adjacent_if_lower_than
    )
    silence_periods = get_silence_periods_in_waveform_frame_scale(
        utterance_periods=utterance_periods, signal=signal
    )
    return utterance_periods, silence_periods


def get_cosine_similarity(x, y):
    x = x + 80
    y = y + 80

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x != 0 and norm_y != 0:
        return np.dot(x, y) / (norm_x * norm_y)
    else:
        return 0


def get_cosine_similarities_along_column(df):
    df_80 = df + 80
    ls = list()
    for (_, col1), (_, col2) in zip(df_80.iteritems(), df_80.loc[:, 1:].iteritems()):
        cos_sim = get_cosine_similarity(col1, col2)
        ls.append(cos_sim)
    return ls


def add_single_value_to_intervals(
    list_of_intervals, value, allows_less_than_or_equal_to
):
    if list_of_intervals:
        popped = list_of_intervals.pop()
        if value - popped[1] <= allows_less_than_or_equal_to:
            list_of_intervals.append((popped[0], value))
        else:
            list_of_intervals.append(popped)
            list_of_intervals.append((value, value))
    else:
        list_of_intervals.append((value, value))
    return list_of_intervals


def get_average_cosine_similarity_of_period(df, period_from, period_to) -> float:
    sum_cos_sim = 0
    for i in range(int(period_from), int(period_to)):
        cos_sim = get_cosine_similarity(df.loc[:, i], df.loc[:, i + 1])
        sum_cos_sim += cos_sim
    if period_from != period_to:
        return sum_cos_sim / (period_to - period_from)
    else:
        return 0


def get_total_duration(periods):
    return sum([row["to"] - row["from"] for _, row in periods.iterrows()])


def get_portions_of_appropriate_amplitude(
    signal,
    period_from,
    period_to,
    MIN_AMP=MIN_AMP,
    MAX_AMP=MAX_AMP
):
    signal_abs = np.abs(signal[period_from: period_to])

    leng = len(signal_abs)
    leng_under = len(
        signal_abs[signal_abs <= MAX_AMP]
    )
    leng_over = len(
        signal_abs[signal_abs >= MIN_AMP]
    )
    return leng_under / leng, leng_over / leng


def get_total_portions_of_appropriate_amplitude(
    signal,
    utterance_periods,
    MIN_AMP=MIN_AMP,
    MAX_AMP=MAX_AMP
) -> Tuple[float, float]:
    sum_leng = 0
    sum_leng_under = 0
    sum_leng_over = 0
    if not utterance_periods.empty:
        for _, row in utterance_periods.iterrows():
            period_from = row["from"]
            period_to = row["to"]
            signal_abs = np.abs(signal[period_from: period_to])

            leng_under = len(
                signal_abs[signal_abs <= MAX_AMP]
            )
            leng_over = len(
                signal_abs[signal_abs >= MIN_AMP]
            )

            sum_leng += (period_to - period_from)
            sum_leng_under += leng_under
            sum_leng_over += leng_over
        portion_under_MAX_AMP = round(sum_leng_under / sum_leng, 3)
        portion_over_MIN_AMP = round(sum_leng_over / sum_leng, 3)
        return portion_under_MAX_AMP, portion_over_MIN_AMP
    else:
        return 0, 0


def get_average_amplitude(df_wav, periods):
    sum_amp = 0
    sum_leng = 0
    for _, row in periods.iterrows():
        y_period = df_wav.loc[row["from"]: row["to"], "signal"]
        y_period_abs = np.abs(y_period)

        sum_amp += np.sum(y_period_abs)
        sum_leng += len(y_period_abs)
    return int(sum_amp / sum_leng)


def get_durations_of_audio(y, sr, utterance_periods) -> Tuple[float, float, float]:
    utterance_periods = _waveform_frame_scale_to_time_scale(
        utterance_periods, sr=sr
    )

    if not utterance_periods.empty:
        duration_silence_bgn = round(utterance_periods.iloc[0, 0], 2)
        tot_duration_utterances = round(
            utterance_periods.iloc[-1, 1] - duration_silence_bgn, 2
        )
        duration_silence_end = round(
            (len(y) / sr) - duration_silence_bgn - tot_duration_utterances, 2
        )
        return duration_silence_bgn, tot_duration_utterances, duration_silence_end
    else:
        return 0, 0, 0


def get_average_noise_value(signal, period_from, period_to):
    signal_silence = signal[period_from: period_to]
    if period_to - period_from != 0:
        return int(np.abs(signal_silence).sum() / (period_to - period_from))
    else:
        return 0


def get_average_noise_value_of_silence_periods(df_wav, silence_periods):
    sum_noise = 0
    sum_leng = 0
    for _, row in silence_periods.iterrows():
        silence_period_from = row["from"]
        silence_period_to = row["to"]

        signal_silence = df_wav.loc[silence_period_from: silence_period_to, "signal"]
        sum_noise += signal_silence.abs().sum()
        sum_leng += silence_period_to - silence_period_from
    if sum_leng != 0:
        return int(round(sum_noise / sum_leng, 0))
    else:
        return 0


def get_number_of_utterance_periods(utterance_periods):
    return len(utterance_periods)


def get_mean_amplitude_by_frequency(df_spec):
    df_spec_80 = df_spec + 80
    mean_amp_by_freq = df_spec_80.mean(axis=1)
    return mean_amp_by_freq


def get_sum_of_amplitude_by_frequency(df_spec):
    df_spec_80 = df_spec + 80
    sum_amp_by_freq = df_spec_80.sum(axis=1)
    return sum_amp_by_freq


def get_unrecorded_frequencies(df_spec):
    sum_amp_by_freq = get_sum_of_amplitude_by_frequency(df_spec)
    ls_freqs = sum_amp_by_freq[sum_amp_by_freq.eq(0)].index.tolist()
    return ls_freqs


def get_period_of_unrecorded_frequencies(df_spec):
    ls_freqs = get_unrecorded_frequencies(df_spec)
    ls_period_freqs = list()
    for f in ls_freqs:
        ls_period_freqs = add_single_value_to_intervals(
            list_of_intervals=ls_period_freqs, value=f, allows_less_than_or_equal_to=50
        )
    return pd.DataFrame(ls_period_freqs, columns=["from", "to"])


def get_medians_along_row(df_spec_db):
    return (df_spec_db + 80).median(axis=0)


def get_plosive_periods(df_spec_db, sr, duration_greater_than=0.05):
    hop_length = int(round(sr * 0.001 * FRAME_SHIFT))

    med = get_medians_along_row(df_spec_db)
    med_nonzero = med[med != 0]
    if med_nonzero.size != 0:
        threshold = np.mean(med_nonzero) * 1.5
    else:
        threshold = 0
    ls = list()
    for i in np.flatnonzero(med > threshold):
        ls = add_single_value_to_intervals(
            list_of_intervals=ls, value=i, allows_less_than_or_equal_to=1
        )
    plosive_periods = pd.DataFrame(ls, columns=["from", "to"])
    plosive_periods = plosive_periods[
        plosive_periods["to"] - plosive_periods["from"]
        > math.ceil(duration_greater_than / hop_length * sr)
    ]
    return plosive_periods[["from", "to"]]


def reduce_noise(signal, sr):
    signal_norm = normalize_signal(signal)

    signal_reduced = nr.reduce_noise(
        y=signal_norm,
        sr=sr,
        # stationary=True,
        # prop_decrease=0.98,
        # time_mask_smooth_ms=50,
        n_fft=int(round(sr * 0.001 * FRAME_LEN)),
        hop_length=int(round(sr * 0.001 * FRAME_SHIFT)),
        n_jobs=-1
    )

    ratio = np.max(signal) / np.max(signal_reduced)
    signal_reduced = (signal_reduced * ratio).astype(np.int16)
    return signal_reduced


def reduce_amplitude_in_silence_periods(signal, silence_periods, ratio=0.2):
    signal_copied = copy.deepcopy(signal)

    for silence_period_from, silence_period_to in silence_periods.values:
        signal_copied[
            silence_period_from: silence_period_to
        ] = signal_copied[
            silence_period_from: silence_period_to
        ] * ratio
    return signal_copied


def gradually_reduce_amplitude_in_silence_periods(signal, silence_periods, n_step=8, splits_into=12) -> np.array:
    signal_copied = copy.deepcopy(signal)

    for idx, (period_from, period_to) in enumerate(silence_periods.values):
        noise_value = get_average_noise_value(signal, period_from, period_to)
        if noise_value != 0:
            ratios = np.linspace(14 / noise_value, 1, n_step)
            leng_subperiod = (period_to - period_from) // splits_into

            if idx == 0:
                signal_copied[period_from: period_to - leng_subperiod] = signal_copied[period_from: period_to - leng_subperiod] * ratios[0]

                for step in range(n_step):
                    subperiod_from = int(period_to - leng_subperiod + leng_subperiod * ((1 / n_step) * step))
                    subperiod_to = int(period_to - leng_subperiod + leng_subperiod * ((1 / n_step) * (step + 1)))
                    signal_copied[subperiod_from: subperiod_to] = signal_copied[subperiod_from: subperiod_to] * ratios[step]
            elif idx == len(silence_periods) - 1:
                signal_copied[period_from + leng_subperiod: period_to] = signal_copied[period_from + leng_subperiod: period_to] * ratios[0]

                for step in range(n_step):
                    subperiod_from = int(period_from + leng_subperiod - leng_subperiod * ((1 / n_step) * (step + 1)))
                    subperiod_to = int(period_from + leng_subperiod - leng_subperiod * ((1 / n_step) * step))
                    signal_copied[subperiod_from: subperiod_to] = signal_copied[subperiod_from: subperiod_to] * ratios[step]
            else:
                signal_copied[period_from + leng_subperiod: period_to - leng_subperiod] = signal_copied[period_from + leng_subperiod: period_to - leng_subperiod] * ratios[0]

                for step in range(n_step):
                    subperiod_from = int(period_to - leng_subperiod + leng_subperiod * ((1 / n_step) * step))
                    subperiod_to = int(period_to - leng_subperiod + leng_subperiod * ((1 / n_step) * (step + 1)))
                    signal_copied[subperiod_from: subperiod_to] = signal_copied[subperiod_from: subperiod_to] * ratios[step]

                    subperiod_from = int(period_from + leng_subperiod - leng_subperiod * ((1 / n_step) * (step + 1)))
                    subperiod_to = int(period_from + leng_subperiod - leng_subperiod * ((1 / n_step) * step))
                    signal_copied[subperiod_from: subperiod_to] = signal_copied[subperiod_from: subperiod_to] * ratios[step]
    return signal_copied


def adjust_amplitude(signal, utterance_periods):
    signal_copied = copy.deepcopy(signal)
    signal_abs = np.abs(signal_copied)

    stand_amp = (5000 + 20000) // 2

    if not utterance_periods.empty:
        for _, row in utterance_periods.iterrows():
            for i in range(row["from"], row["to"]):
                tar = signal_abs[i]
                if tar - stand_amp >= 7500:
                    value = 7500 / (tar - stand_amp)
                    signal_copied[i: i + 1] = signal_copied[i: i + 1] * value
                elif stand_amp - tar >= 7500:
                    value = (stand_amp - tar) / 5000
                    signal_copied[i: i + 1] = signal_copied[i: i + 1] * value
    return signal_copied
