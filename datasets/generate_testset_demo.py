"""Generate (or regenerate) datasets/wiki_50.json — the Phase 2 demo dataset.

This script is intentionally network-free and LLM-free: it writes hand-crafted
QA pairs directly. For RAGAS-powered generation see generate_testset.py (Phase 3).

Usage:
    python datasets/generate_testset_demo.py
    python datasets/generate_testset_demo.py --out datasets/my_custom.json
    python datasets/generate_testset_demo.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hand-crafted QA pairs — same data as wiki_50.json
# Kept here so the dataset is always reproducible from source.
# ---------------------------------------------------------------------------

RAW_PAIRS: list[dict] = [
    {
        "id": "wiki_001",
        "question": "In what year did the Berlin Wall fall?",
        "ground_truth": "1989",
        "reference_contexts": [
            "The Berlin Wall, constructed in 1961 on the orders of the East German government, divided East and West Berlin for 28 years. It fell on 9 November 1989 when East Germany opened its borders following mass protests."
        ],
    },
    {
        "id": "wiki_002",
        "question": "What is the chemical symbol for gold?",
        "ground_truth": "Au",
        "reference_contexts": [
            "Gold is a chemical element with the symbol Au (from the Latin aurum) and atomic number 79. It is a bright, slightly orange-yellow, dense, soft, malleable, and ductile metal."
        ],
    },
    {
        "id": "wiki_003",
        "question": "Who wrote the play Hamlet?",
        "ground_truth": "William Shakespeare",
        "reference_contexts": [
            "Hamlet, Prince of Denmark is a tragedy written by William Shakespeare, believed to have been written between 1599 and 1601."
        ],
    },
    {
        "id": "wiki_004",
        "question": "What is the capital city of Japan?",
        "ground_truth": "Tokyo",
        "reference_contexts": [
            "Tokyo, officially the Tokyo Metropolis, is the capital and most populous city of Japan. It has been the seat of the Japanese government since 1869."
        ],
    },
    {
        "id": "wiki_005",
        "question": "What planet is known as the Red Planet?",
        "ground_truth": "Mars",
        "reference_contexts": [
            "Mars is the fourth planet from the Sun and is often referred to as the Red Planet due to its reddish appearance caused by iron oxide (rust) on its surface."
        ],
    },
    {
        "id": "wiki_006",
        "question": "In what year did World War II end?",
        "ground_truth": "1945",
        "reference_contexts": [
            "World War II lasted from 1939 to 1945. It ended in Europe on 8 May 1945 (V-E Day) and in the Pacific on 15 August 1945 (V-J Day) following Japan's surrender."
        ],
    },
    {
        "id": "wiki_007",
        "question": "What is the speed of light in a vacuum, approximately?",
        "ground_truth": "approximately 299,792 kilometres per second",
        "reference_contexts": [
            "The speed of light in a vacuum, commonly denoted c, is approximately 299,792 kilometres per second (km/s). It is a fundamental physical constant and the ultimate speed limit of the universe."
        ],
    },
    {
        "id": "wiki_008",
        "question": "Which scientist proposed the theory of general relativity?",
        "ground_truth": "Albert Einstein",
        "reference_contexts": [
            "The general theory of relativity was published by Albert Einstein in 1915. It describes gravity as a geometric property of space and time."
        ],
    },
    {
        "id": "wiki_009",
        "question": "What is the largest ocean on Earth?",
        "ground_truth": "Pacific Ocean",
        "reference_contexts": [
            "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It covers more than 165 million square kilometres."
        ],
    },
    {
        "id": "wiki_010",
        "question": "How many bones are in the adult human body?",
        "ground_truth": "206",
        "reference_contexts": [
            "The adult human skeleton consists of 206 bones. Babies are born with approximately 270 to 300 bones; many fuse together during childhood and adolescence."
        ],
    },
    {
        "id": "wiki_011",
        "question": "What is the hardest natural substance on Earth?",
        "ground_truth": "diamond",
        "reference_contexts": [
            "Diamond is the hardest known natural material on Earth, scoring 10 on the Mohs hardness scale. It is a form of carbon where the atoms are arranged in a crystal structure called diamond cubic."
        ],
    },
    {
        "id": "wiki_012",
        "question": "Who painted the Mona Lisa?",
        "ground_truth": "Leonardo da Vinci",
        "reference_contexts": [
            "The Mona Lisa is a half-length portrait painting by the Italian Renaissance polymath Leonardo da Vinci, believed to have been painted between 1503 and 1519."
        ],
    },
    {
        "id": "wiki_013",
        "question": "What is the smallest country in the world by area?",
        "ground_truth": "Vatican City",
        "reference_contexts": [
            "Vatican City is an independent city-state enclaved within Rome, Italy. With an area of approximately 44 hectares (110 acres), it is the smallest country in the world both by area and by population."
        ],
    },
    {
        "id": "wiki_014",
        "question": "What is the most abundant gas in Earth's atmosphere?",
        "ground_truth": "nitrogen",
        "reference_contexts": [
            "Earth's atmosphere is composed of approximately 78% nitrogen, 21% oxygen, and 1% other gases. Nitrogen (N2) is therefore the most abundant gas in the atmosphere."
        ],
    },
    {
        "id": "wiki_015",
        "question": "Who was the first person to walk on the Moon?",
        "ground_truth": "Neil Armstrong",
        "reference_contexts": [
            "Neil Armstrong became the first person to walk on the Moon on 20 July 1969, during NASA's Apollo 11 mission."
        ],
    },
    {
        "id": "wiki_016",
        "question": "What is the longest river in the world?",
        "ground_truth": "Nile",
        "reference_contexts": [
            "The Nile is a major north-flowing river in northeastern Africa. At approximately 6,650 kilometres (4,130 miles) in length, it is widely considered the longest river in the world."
        ],
    },
    {
        "id": "wiki_017",
        "question": "In which country is the Great Wall located?",
        "ground_truth": "China",
        "reference_contexts": [
            "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states and Imperial China."
        ],
    },
    {
        "id": "wiki_018",
        "question": "What element has the atomic number 1?",
        "ground_truth": "hydrogen",
        "reference_contexts": [
            "Hydrogen is the chemical element with the symbol H and atomic number 1. It is the lightest element in the periodic table and the most abundant chemical substance in the universe."
        ],
    },
    {
        "id": "wiki_019",
        "question": "What is the capital of Australia?",
        "ground_truth": "Canberra",
        "reference_contexts": [
            "Canberra is the capital city of Australia, situated within the Australian Capital Territory (ACT). The city became the national capital in 1913."
        ],
    },
    {
        "id": "wiki_020",
        "question": "How many continents are there on Earth?",
        "ground_truth": "7",
        "reference_contexts": [
            "There are seven continents on Earth: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America."
        ],
    },
    {
        "id": "wiki_021",
        "question": "What is the chemical formula for water?",
        "ground_truth": "H2O",
        "reference_contexts": [
            "Water is an inorganic compound with the chemical formula H2O. It consists of two hydrogen atoms covalently bonded to one oxygen atom."
        ],
    },
    {
        "id": "wiki_022",
        "question": "Who invented the telephone?",
        "ground_truth": "Alexander Graham Bell",
        "reference_contexts": [
            "The telephone was invented by Alexander Graham Bell, who was awarded the first patent for it on 7 March 1876."
        ],
    },
    {
        "id": "wiki_023",
        "question": "What is the tallest mountain in the world?",
        "ground_truth": "Mount Everest",
        "reference_contexts": [
            "Mount Everest, at 8,848.86 metres above sea level, is Earth's highest mountain. It is located in the Mahalangur Himal sub-range of the Himalayas on the border of Nepal and Tibet."
        ],
    },
    {
        "id": "wiki_024",
        "question": "What language has the most native speakers in the world?",
        "ground_truth": "Mandarin Chinese",
        "reference_contexts": [
            "Mandarin Chinese has the most native speakers of any language in the world, with approximately 920 million native speakers."
        ],
    },
    {
        "id": "wiki_025",
        "question": "What is the powerhouse of the cell?",
        "ground_truth": "mitochondria",
        "reference_contexts": [
            "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of ATP, which is why they are often referred to as the powerhouse of the cell."
        ],
    },
    {
        "id": "wiki_026",
        "question": "Who wrote the novel '1984'?",
        "ground_truth": "George Orwell",
        "reference_contexts": [
            "Nineteen Eighty-Four (commonly referred to as 1984) is a dystopian novel written by George Orwell, published on 8 June 1949."
        ],
    },
    {
        "id": "wiki_027",
        "question": "What is the currency of the United Kingdom?",
        "ground_truth": "pound sterling",
        "reference_contexts": [
            "The pound sterling, commonly known as the pound, is the official currency of the United Kingdom."
        ],
    },
    {
        "id": "wiki_028",
        "question": "What is the boiling point of water at standard atmospheric pressure?",
        "ground_truth": "100 degrees Celsius (212 degrees Fahrenheit)",
        "reference_contexts": [
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm or 101.325 kPa)."
        ],
    },
    {
        "id": "wiki_029",
        "question": "Which planet has the most moons in the Solar System?",
        "ground_truth": "Saturn",
        "reference_contexts": [
            "Saturn, the sixth planet from the Sun, holds the record for the most confirmed moons in the Solar System with 146 known moons as of 2023."
        ],
    },
    {
        "id": "wiki_030",
        "question": "What is the Pythagorean theorem?",
        "ground_truth": "In a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides (a² + b² = c²)",
        "reference_contexts": [
            "The Pythagorean theorem states that in a right triangle the area of the square on the hypotenuse is equal to the sum of the areas of the squares on the other two sides, expressed as a² + b² = c²."
        ],
    },
    {
        "id": "wiki_031",
        "question": "Who is credited with discovering penicillin?",
        "ground_truth": "Alexander Fleming",
        "reference_contexts": [
            "Penicillin was discovered by Scottish bacteriologist Alexander Fleming in 1928, when he noticed a mould (Penicillium notatum) killing surrounding bacteria in a contaminated petri dish."
        ],
    },
    {
        "id": "wiki_032",
        "question": "What is the largest planet in the Solar System?",
        "ground_truth": "Jupiter",
        "reference_contexts": [
            "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than 2.5 times that of all other planets combined."
        ],
    },
    {
        "id": "wiki_033",
        "question": "In what year was the World Wide Web invented?",
        "ground_truth": "1989",
        "reference_contexts": [
            "The World Wide Web was invented by British computer scientist Tim Berners-Lee in 1989 while working at CERN."
        ],
    },
    {
        "id": "wiki_034",
        "question": "What is the largest organ in the human body?",
        "ground_truth": "skin",
        "reference_contexts": [
            "The skin is the largest organ of the human body, with a total area of about 20 square feet in adults. Its functions include protecting the body from pathogens and regulating temperature."
        ],
    },
    {
        "id": "wiki_035",
        "question": "Which country invented paper?",
        "ground_truth": "China",
        "reference_contexts": [
            "Paper was invented in China during the Han dynasty around 105 AD, traditionally attributed to the court official Cai Lun."
        ],
    },
    {
        "id": "wiki_036",
        "question": "What is the freezing point of water in Fahrenheit?",
        "ground_truth": "32 degrees Fahrenheit",
        "reference_contexts": [
            "Water freezes at 0 degrees Celsius, which corresponds to 32 degrees Fahrenheit, at standard atmospheric pressure."
        ],
    },
    {
        "id": "wiki_037",
        "question": "Who developed the first theory of evolution by natural selection?",
        "ground_truth": "Charles Darwin",
        "reference_contexts": [
            "Charles Darwin published his theory of evolution by natural selection in his 1859 work 'On the Origin of Species'."
        ],
    },
    {
        "id": "wiki_038",
        "question": "What is the capital of Brazil?",
        "ground_truth": "Brasília",
        "reference_contexts": [
            "Brasília is the federal capital of Brazil. The city was purpose-built and inaugurated as the new national capital on 21 April 1960, replacing the former capital Rio de Janeiro."
        ],
    },
    {
        "id": "wiki_039",
        "question": "How many planets are in the Solar System?",
        "ground_truth": "8",
        "reference_contexts": [
            "There are eight recognized planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006."
        ],
    },
    {
        "id": "wiki_040",
        "question": "What is the national language of Brazil?",
        "ground_truth": "Portuguese",
        "reference_contexts": [
            "Portuguese is the official and national language of Brazil, making it the world's largest Portuguese-speaking nation."
        ],
    },
    {
        "id": "wiki_041",
        "question": "What structure connects the two cerebral hemispheres?",
        "ground_truth": "corpus callosum",
        "reference_contexts": [
            "The corpus callosum is a wide flat bundle of neural fibres beneath the cerebral cortex that connects the left and right cerebral hemispheres and facilitates communication between them."
        ],
    },
    {
        "id": "wiki_042",
        "question": "Who was the first President of the United States?",
        "ground_truth": "George Washington",
        "reference_contexts": [
            "George Washington served as the first President of the United States from 1789 to 1797. He was unanimously elected by the Electoral College."
        ],
    },
    {
        "id": "wiki_043",
        "question": "What is the main ingredient of traditional Japanese miso soup?",
        "ground_truth": "miso paste (fermented soybean paste)",
        "reference_contexts": [
            "Miso soup is a traditional Japanese soup made with a dashi stock and softened miso paste. Miso paste, made from fermented soybeans, gives the soup its characteristic umami flavour."
        ],
    },
    {
        "id": "wiki_044",
        "question": "In what city was the Eiffel Tower built?",
        "ground_truth": "Paris",
        "reference_contexts": [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, designed by Gustave Eiffel for the 1889 World's Fair."
        ],
    },
    {
        "id": "wiki_045",
        "question": "What is the process by which plants make food using sunlight?",
        "ground_truth": "photosynthesis",
        "reference_contexts": [
            "Photosynthesis is the process by which plants, algae, and some bacteria use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
        ],
    },
    {
        "id": "wiki_046",
        "question": "Which ocean is the smallest on Earth?",
        "ground_truth": "Arctic Ocean",
        "reference_contexts": [
            "The Arctic Ocean is the smallest and shallowest of the world's five major oceans, covering approximately 14,060,000 square kilometres."
        ],
    },
    {
        "id": "wiki_047",
        "question": "What is the most widely spoken language in the world by total speakers?",
        "ground_truth": "English",
        "reference_contexts": [
            "English is the most widely spoken language in the world when counting both native and non-native speakers, with an estimated 1.5 billion total speakers."
        ],
    },
    {
        "id": "wiki_048",
        "question": "What does DNA stand for?",
        "ground_truth": "deoxyribonucleic acid",
        "reference_contexts": [
            "DNA stands for deoxyribonucleic acid. It is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms."
        ],
    },
    {
        "id": "wiki_049",
        "question": "What is the world's largest desert by area?",
        "ground_truth": "Antarctic Desert",
        "reference_contexts": [
            "The Antarctic Desert is the world's largest desert, covering the continent of Antarctica with an area of about 14.2 million square kilometres. Antarctica receives very little precipitation, making it technically a cold desert."
        ],
    },
    {
        "id": "wiki_050",
        "question": "What is the name of the force that keeps planets in orbit around the Sun?",
        "ground_truth": "gravity",
        "reference_contexts": [
            "Gravity is the fundamental force that keeps planets in orbit around the Sun. Isaac Newton formulated the law of universal gravitation describing how every particle of matter attracts every other particle."
        ],
    },
]


def _add_null_tools(records: list[dict]) -> list[dict]:
    return [{**r, "expected_tools": None} for r in records]


def _validate(records: list[dict]) -> list[str]:
    errors: list[str] = []
    ids_seen: set[str] = set()
    for r in records:
        rid = r.get("id", "<missing>")
        if rid in ids_seen:
            errors.append(f"Duplicate id: {rid}")
        ids_seen.add(rid)
        if not r.get("question"):
            errors.append(f"{rid}: missing question")
        if not r.get("ground_truth"):
            errors.append(f"{rid}: missing ground_truth")
        ctxs = r.get("reference_contexts", [])
        if not ctxs or not all(ctxs):
            errors.append(f"{rid}: reference_contexts empty or contains blank entry")
    if len(records) != 50:
        errors.append(f"Expected 50 records, got {len(records)}")
    return errors


def generate(out_path: str) -> None:
    records = _add_null_tools(RAW_PAIRS)
    errors = _validate(records)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    Path(out_path).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate wiki_50.json demo dataset")
    parser.add_argument("--out", default="datasets/wiki_50.json", help="Output path")
    parser.add_argument("--validate-only", action="store_true", help="Validate without writing")
    args = parser.parse_args(argv)

    records = _add_null_tools(RAW_PAIRS)
    errors = _validate(records)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print(f"OK — {len(records)} records pass validation")
        return

    generate(args.out)


if __name__ == "__main__":
    main()
