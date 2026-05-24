LABELS = [
    "bass",
    "cello",
    "clarinet",
    "drums",
    "flute",
    "gac",
    "gel",
    "organ",
    "piano",
    "saxophone",
    "trumpet",
    "violin",
    "voice"
]

LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)