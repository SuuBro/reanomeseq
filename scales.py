import math

scale_intervals = {
    "blues": [3, 2, 1, 1, 3, 2],
    "dorian": [2, 1, 2, 2, 2, 1, 2],
    "major": [2, 2, 1, 2, 2, 2, 1],
    "minor": [2, 1, 2, 2, 1, 2, 2],
    "locrian": [1, 2, 2, 1, 2, 2, 2],
    "mixolydian": [2, 2, 1, 2, 2, 1, 2],
    "phrygian": [1, 2, 2, 2, 1, 2, 2],
}

names_from_interval = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}

"""From an interval give the note name, favouring sharps over flats."""
interval_from_names = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0
}

class Note:
    """A single note in a given octave, e.g. C#3.
    Measured as a number of semitones above Middle C:
        * Note(0) # Middle C, i.e. C3
        * Note(2) # D3
    """

    semitones_above_middle_c: int
    name: str

    def __init__(self, name: str = None, semitones_above_middle_c: int = None):
        """Create a note with a given name or degree.
        Examples:
            * Note("C#")
            * Note(semitones_above_middle_c = 1)
        """
        if name is not None:
            if name not in interval_from_names:
                raise Exception(f"No note found with name {name}.")
            self._set_degree(interval_from_names[name])
        elif semitones_above_middle_c is not None:
            self._set_degree(semitones_above_middle_c)
        else:
            self._set_degree(0)

    def _set_degree(self, semitones_above_middle_c: int):
        """Set the note name and octave.
        Should only be used during initialisation.
        """
        self.semitones_above_middle_c = semitones_above_middle_c
        self.name = names_from_interval[semitones_above_middle_c % 12]

    def __str__(self):
        return self.midi

    def __repr__(self):
        return self.midi

    @property
    def midi(self):
        return self.semitones_above_middle_c

    def __add__(self, shift: int):
        """Shifting this note's degree upwards."""
        return Note(semitones_above_middle_c=self.semitones_above_middle_c + shift)

    def __sub__(self, shift: int):
        """Shifting this note's degree downwards."""
        return self + (-shift)

    def __eq__(self, other):
        """Check equality via .midi."""
        if isinstance(other, Note):
            return self.midi == other.midi
        else:
            return self.midi == other or self.name == other

def scale(starting_note, mode="major"):
    """Return a sequence of Notes starting on the given note in the given mode.
    Example:
        * scale("C") # C major (ionian)
        * scale(Note(4), "harmonic minor") # E harmonic minor
    """
    if mode not in scale_intervals:
        raise Exception(f"The mode {mode} is not available.")
    if not isinstance(starting_note, Note):
        starting_note = Note(starting_note)
    notes = [starting_note]
    for interval in scale_intervals[mode]:
        notes.append(notes[-1] + interval)
    return notes[:len(scale_intervals[mode])]