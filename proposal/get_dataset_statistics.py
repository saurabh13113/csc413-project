import numpy as np
import matplotlib.pyplot as plt
from music21 import corpus, pitch


def get_stats():
    chorales = corpus.chorales.Iterator(returnType="filename")

    chorale_lengths = []
    pitch_notes = {
        "Soprano": [],
        "Alto": [],
        "Tenor": [],
        "Bass": [],
    }

    major_count = 0
    minor_count = 0


    for chorale in chorales:
        score = corpus.parse(chorale)
        duration = score.highestTime
        chorale_lengths.append(round(duration * 4))

        if score.analyze("key").mode == "major":
            major_count += 1
        else:
            minor_count += 1

        parts = score.parts
        if len(parts) >= 4:
            for voice, part in zip(pitch_notes.keys(), parts[:4]):
                for note in part.flatten().notes:
                    if note.isNote:
                        pitch_notes[voice].append(note.pitch.midi)


    pitch_ranges = {}
    for voice, pitches in pitch_notes.items():
        pitch_ranges[voice] = (pitch.Pitch(min(pitches)).nameWithOctave, pitch.Pitch(max(pitches)).nameWithOctave)


    total_keys = major_count + minor_count
    total_notes = sum(len(note_array) for note_array in pitch_notes.values())
    mean_chorale_length = np.mean(chorale_lengths)
    max_chorale_length = np.max(chorale_lengths)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(chorale_lengths, bins=30, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Chorale length (sixteenth-note steps)")
    ax1.set_ylabel("Number of chorales")
    ax1.set_title("Distribution of Chorale Lengths")
    ax1.grid(axis="y", alpha=0.25)

    colors = {
        "Soprano": "tab:blue",
        "Alto": "tab:orange",
        "Tenor": "tab:green",
        "Bass": "tab:red",
    }

    all_pitches = [
        pitch
        for pitches in pitch_notes.values()
        for pitch in pitches
    ]

    pitch_bins = np.arange(min(all_pitches) - 0.5, max(all_pitches) + 1.5, 1)

    bottom = np.zeros(len(pitch_bins) - 1)

    for voice, pitches in pitch_notes.items():
        counts, _ = np.histogram(pitches, bins=pitch_bins)
        ax2.bar(
            pitch_bins[:-1],
            counts,
            width=1,
            bottom=bottom,
            color=colors[voice],
            label=voice,
            align="edge",
            edgecolor="white",
            linewidth=0.3,
        )

        bottom += counts

    ax2.set_xlabel("MIDI pitch")
    ax2.set_ylabel("Number of notes")
    ax2.set_title("Pitch Distribution by Voice")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig("bach_chorale_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()



    print(f"Num chorales: {len(chorale_lengths)}")
    print(f"Total notes: {total_notes}")
    print(f"Mean chorale length: {mean_chorale_length:.2f}")
    print(f"Max chorale length: {max_chorale_length}")

    print("\nPitch ranges:")
    for voice, (low, high) in pitch_ranges.items():
        print(f"  {voice}: {low} - {high}")

    print("\nKey proportions:")
    print(f"  Major: {major_count}/{total_keys} ({major_count / total_keys:.1%})")
    print(f"  Minor: {minor_count}/{total_keys} ({minor_count / total_keys:.1%})")


if __name__ == "__main__":
    get_stats()