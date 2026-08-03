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
    name: "狂欢节 → 海洋",
    category: "多跳推理",
    prompt: "Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the ",
    intermediate: "Brazil",
    target: "Atlantic",
  },
  {
    id: "amazon-language",
    name: "亚马逊河 → 语言",
    category: "多跳推理",
    prompt: "Fact: The language spoken in the country where the Amazon River ends is ",
    intermediate: "Brazil",
    target: "Portuguese",
  },
  {
    id: "mars-color",
    name: "行星 → 颜色",
    category: "多跳推理",
    prompt: "Fact: The color of the planet fourth from the Sun is ",
    intermediate: "Mars",
    target: "red",
  },
];
