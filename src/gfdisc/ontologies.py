"""Phase-2 target ontologies -- pre-registered before any result is inspected.

Each entry declares the structure expected if the model represents the concept
geometrically, so the prediction can fail visibly rather than being fitted after
the fact. The musical-keys entry is the sharpest: circle-of-fifths order and
chromatic order are different permutations of the same twelve tokens, so scoring
against both distinguishes a real finding from a generic "these tokens cluster".
"""

CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

DIRECTIONS = ["north", "northeast", "east", "southeast",
              "south", "southwest", "west", "northwest"]

# Ordinal, not cyclic: predicts a 1-D curve. A clean negative on the circle test
# here is informative rather than embarrassing -- it is the contrast case.
MILITARY_RANKS = ["private", "corporal", "sergeant", "lieutenant", "captain",
                  "major", "colonel", "general"]

# Kinship predicts a 2-D lattice (generation x lineality), not a ring, so it
# exercises a capability Phase 1 never tested.
KINSHIP = ["grandfather", "grandmother", "father", "mother", "son", "daughter",
           "brother", "sister", "uncle", "aunt", "nephew", "niece", "cousin"]
KINSHIP_GENERATION = [2, 2, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0]
KINSHIP_LINEAL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

ONTOLOGIES = {
    "fifths":     dict(words=CIRCLE_OF_FIFTHS, kind="cycle", k=12,
                       rival=dict(name="chromatic", words=CHROMATIC)),
    "zodiac":     dict(words=ZODIAC, kind="cycle", k=12),
    "directions": dict(words=DIRECTIONS, kind="cycle", k=8),
    "ranks":      dict(words=MILITARY_RANKS, kind="ordinal", k=8),
    "kinship":    dict(words=KINSHIP, kind="lattice", k=13,
                       axes=dict(generation=KINSHIP_GENERATION, lineal=KINSHIP_LINEAL)),
}
