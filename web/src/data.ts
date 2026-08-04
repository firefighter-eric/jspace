export type Sample = {
  id: string;
  name: string;
  category: string;
  prompt: string;
  intermediate: string;
  target: string;
};

export const samples: Sample[] = [
  {
    id: "carnival-ocean",
    name: "Carnival",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the ",
    intermediate: "Brazil",
    target: "Atlantic",
  },
  {
    id: "amazon-language",
    name: "Amazon River",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The language spoken in the country where the Amazon River ends is ",
    intermediate: "Brazil",
    target: "Portuguese",
  },
  {
    id: "mars-color",
    name: "Fourth planet",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The color of the planet fourth from the Sun is ",
    intermediate: "Mars",
    target: "red",
  },
  {
    id: "spider-legs",
    name: "Spider legs",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The number of legs on the animal that spins webs is ",
    intermediate: "spider",
    target: "8",
  },
  {
    id: "basketball-players",
    name: "Basketball side",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The number of players per side in the sport invented in Springfield, Massachusetts is ",
    intermediate: "basketball",
    target: "5",
  },
  {
    id: "paper-continent",
    name: "Paper invention",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The continent where the country that invented paper is located is ",
    intermediate: "China",
    target: "Asia",
  },
  {
    id: "christmas-season",
    name: "Christmas season",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The season when the holiday with a decorated tree occurs is ",
    intermediate: "Christmas",
    target: "winter",
  },
  {
    id: "osu-rival-mascot",
    name: "Ohio State rival",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The mascot of the college football rival of Ohio State is a ",
    intermediate: "Michigan",
    target: "wolverine",
  },
  {
    id: "topeka-west",
    name: "West of Topeka",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The state west of the state with Topeka as its capital is ",
    intermediate: "Kansas",
    target: "Colorado",
  },
  {
    id: "atomic-79-symbol",
    name: "Element 79",
    category: "Anthropic eval · multihop",
    prompt: "Fact: The chemical symbol for the element with atomic number 79 is ",
    intermediate: "gold",
    target: "Au",
  },
];
