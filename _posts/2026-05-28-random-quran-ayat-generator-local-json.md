---
layout: post
title: "آیه تصادفی قرآن"
date: 2026-05-28
categories: [javascript, web, quran]
tags: [html, css, javascript, json, quran, farsi]
---

این پروژه یک آیه تصادفی قرآن را همراه با ترجمه فارسی نشان می‌دهد.

انتخاب آیه با یک روش وزن‌دهی شده بر اساس عدد ۱۹ انجام می‌شود.

<div class="quran-container">
  <div class="quoteBox">
    <h1 class="quran-title">آیه تصادفی قرآن</h1>

    <div id="content">
      <div id="arabicVerseText">در حال بارگذاری...</div>
      <div id="verseText" class="mediumSize"></div>
      <div id="surahAndAyah" class="mediumSize"></div>
      <div id="randomMethod"></div>
      <div id="sourceNote"></div>
      <div id="errorMessage"></div>
    </div>

    <input id="shuffle" type="button" value="آیه جدید">
  </div>
</div>

<style>
@font-face {
  font-family: "UthmanicHafs";
  src: url('{{ "/assets/fonts/UthmanicHafs1%20Ver16.ttf" | relative_url }}') format("truetype");
  font-weight: normal;
  font-style: normal;
}

.quran-container {
  text-align: center;
  border-radius: 18px;
  position: relative;
  margin: 15px auto;
  width: 85%;
  max-width: 950px;
  background-color: transparent;
}

.quoteBox {
  border-radius: 22px;
  position: relative;
  margin: 15px auto;
  padding: 45px 45px;
  min-height: 620px;
  background-image: url('{{ "/assets/images/quran-background.jpeg" | relative_url }}');
  background-size: cover;
  background-position: center center;
  background-repeat: no-repeat;
  border: 2px solid rgba(46, 125, 50, 0.55);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.quran-title {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  font-size: 120%;
  font-weight: normal;
  direction: rtl;
  text-align: center;
  margin-bottom: 25px;
  color: #145a32;
}

#content {
  margin: 0 auto 30px auto;
  max-width: 820px;
  background: transparent;
  border: none;
  border-radius: 18px;
  padding: 24px 22px;
  box-shadow: none;
}

#arabicVerseText {
  font-family: "UthmanicHafs", "Amiri", "Scheherazade New", "Traditional Arabic", serif;
  font-size: 220%;
  direction: rtl;
  text-align: center;
  margin: 20px 0;
  line-height: 2.1;
  color: #0b3d1c;
  text-shadow: 0 1px 3px rgba(255, 255, 255, 0.75);
}

#verseText {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  direction: rtl;
  text-align: center;
  unicode-bidi: plaintext;
  color: #1b5e20;
  text-shadow: 0 1px 3px rgba(255, 255, 255, 0.75);
}

.mediumSize {
  font-size: 135%;
  margin: 16px 0;
  line-height: 1.9;
}

#randomMethod {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  font-size: 95%;
  color: #666;
  margin-top: 18px;
  direction: rtl;
  text-align: center;
}

#sourceNote {
  font-family: Tahoma, Arial, sans-serif;
  font-size: 80%;
  color: #777;
  margin-top: 12px;
  direction: rtl;
  text-align: center;
}

#sourceNote a {
  color: #555;
}

#errorMessage {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  font-size: 110%;
  color: #b00020;
  margin-top: 20px;
}

input[type=button] {
  cursor: pointer;
  padding: 10px 22px;
  margin: 8px;
  border: none;
  border-radius: 5px;
  background-color: #1192d3;
  color: white;
  font-size: 15px;
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
}

input[type=button]:hover {
  background-color: #0d75aa;
}

@media (max-width: 650px) {
  .quran-container {
    width: 92%;
  }

  .quoteBox {
    padding: 25px 18px;
    min-height: 520px;
    background-position: center center;
  }

  .quran-title {
    font-size: 110%;
  }

  #arabicVerseText {
    font-size: 185%;
  }

  .mediumSize {
    font-size: 120%;
  }

  #randomMethod {
    font-size: 85%;
  }
}
</style>

<script>
let quranAyat = [];
let arText = "";
let faText = "";
let surahAndAyah = "";

document.addEventListener("DOMContentLoaded", async function () {
  await loadQuranData();

  document.getElementById("shuffle").addEventListener("click", showRandomAyah);
});

async function loadQuranData() {
  try {
    const dataUrl = "{{ '/assets/data/quran_fa_ansarian.json' | relative_url }}";
    const response = await fetch(dataUrl);

    if (!response.ok) {
      throw new Error("Local Quran JSON file could not be loaded.");
    }

    quranAyat = await response.json();

    if (!Array.isArray(quranAyat) || quranAyat.length === 0) {
      throw new Error("Quran JSON file is empty or invalid.");
    }

    showRandomAyah();

  } catch (error) {
    document.getElementById("arabicVerseText").textContent = "";
    document.getElementById("verseText").textContent = "";
    document.getElementById("surahAndAyah").textContent = "";
    document.getElementById("randomMethod").textContent = "";
    document.getElementById("sourceNote").textContent = "";
    document.getElementById("errorMessage").textContent =
      "فایل قرآن پیدا نشد. مطمئن شوید quran_fa_ansarian.json داخل assets/data قرار دارد.";

    console.error(error);
  }
}

function fixFarsiBrackets(text) {
  return text
    .replace(/\[/g, "TEMP_OPEN_BRACKET")
    .replace(/\]/g, "«")
    .replace(/TEMP_OPEN_BRACKET/g, "»")
    .replace(/([^\s])«/g, "$1 «")
    .replace(/»([^\s،؛:.!؟])/g, "» $1");
}

function numberToPersianWords(n) {
  const ones = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"];

  const teens = {
    10: "ده",
    11: "یازده",
    12: "دوازده",
    13: "سیزده",
    14: "چهارده",
    15: "پانزده",
    16: "شانزده",
    17: "هفده",
    18: "هجده",
    19: "نوزده"
  };

  const tens = {
    20: "بیست",
    30: "سی",
    40: "چهل",
    50: "پنجاه",
    60: "شصت",
    70: "هفتاد",
    80: "هشتاد",
    90: "نود"
  };

  const hundreds = {
    100: "صد",
    200: "دویست",
    300: "سیصد",
    400: "چهارصد",
    500: "پانصد",
    600: "ششصد",
    700: "هفتصد",
    800: "هشتصد",
    900: "نهصد"
  };

  if (n < 10) {
    return ones[n];
  }

  if (n < 20) {
    return teens[n];
  }

  if (n < 100) {
    const t = Math.floor(n / 10) * 10;
    const r = n % 10;
    return r === 0 ? tens[t] : tens[t] + " و " + ones[r];
  }

  if (n < 1000) {
    const h = Math.floor(n / 100) * 100;
    const r = n % 100;
    return r === 0 ? hundreds[h] : hundreds[h] + " و " + numberToPersianWords(r);
  }

  const th = Math.floor(n / 1000);
  const r = n % 1000;
  const thText = th === 1 ? "هزار" : numberToPersianWords(th) + " هزار";

  return r === 0 ? thText : thText + " و " + numberToPersianWords(r);
}

function toPersianOrdinal(n) {
  const ordinalWords = {
    "یک": "اولین",
    "دو": "دومین",
    "سه": "سومین",
    "چهار": "چهارمین",
    "پنج": "پنجمین",
    "شش": "ششمین",
    "هفت": "هفتمین",
    "هشت": "هشتمین",
    "نه": "نهمین",
    "ده": "دهمین",
    "یازده": "یازدهمین",
    "دوازده": "دوازدهمین",
    "سیزده": "سیزدهمین",
    "چهارده": "چهاردهمین",
    "پانزده": "پانزدهمین",
    "شانزده": "شانزدهمین",
    "هفده": "هفدهمین",
    "هجده": "هجدهمین",
    "نوزده": "نوزدهمین",
    "بیست": "بیستمین",
    "سی": "سی‌امین",
    "چهل": "چهلمین",
    "پنجاه": "پنجاهمین",
    "شصت": "شصتمین",
    "هفتاد": "هفتادمین",
    "هشتاد": "هشتادمین",
    "نود": "نودمین",
    "صد": "صدمین",
    "دویست": "دویستمین",
    "سیصد": "سیصدمین",
    "چهارصد": "چهارصدمین",
    "پانصد": "پانصدمین",
    "ششصد": "ششصدمین",
    "هفتصد": "هفتصدمین",
    "هشتصد": "هشتصدمین",
    "نهصد": "نهصدمین",
    "هزار": "هزارمین"
  };

  const words = numberToPersianWords(n);
  const parts = words.split(" و ");
  const lastPart = parts[parts.length - 1];

  if (ordinalWords[lastPart]) {
    parts[parts.length - 1] = ordinalWords[lastPart];
    return parts.join(" و ");
  }

  return words + "مین";
}

function getAdvanced19RandomAyah() {
  if (!quranAyat.length) {
    return null;
  }

  const candidates = quranAyat.map(function (ayah) {
    let weight = 1;

    if (ayah.globalAyah % 19 === 0) {
      weight += 10;
    }

    if (ayah.surahNumber % 19 === 0) {
      weight += 6;
    }

    if (ayah.ayahNumber % 19 === 0) {
      weight += 6;
    }

    if ((ayah.surahNumber + ayah.ayahNumber) % 19 === 0) {
      weight += 5;
    }

    if ((ayah.surahNumber * ayah.ayahNumber) % 19 === 0) {
      weight += 5;
    }

    if (String(ayah.globalAyah).includes("19")) {
      weight += 4;
    }

    const digitSum = String(ayah.globalAyah)
      .split("")
      .map(Number)
      .reduce(function (a, b) {
        return a + b;
      }, 0);

    if (digitSum % 19 === 0) {
      weight += 3;
    }

    return {
      ayah: ayah,
      weight: weight
    };
  });

  const totalWeight = candidates.reduce(function (sum, item) {
    return sum + item.weight;
  }, 0);

  let randomValue = Math.random() * totalWeight;

  for (const item of candidates) {
    randomValue -= item.weight;

    if (randomValue <= 0) {
      return item.ayah;
    }
  }

  return quranAyat[Math.floor(Math.random() * quranAyat.length)];
}

function showRandomAyah() {
  document.getElementById("errorMessage").textContent = "";

  const ayah = getAdvanced19RandomAyah();

  if (!ayah) {
    document.getElementById("arabicVerseText").textContent = "در حال بارگذاری...";
    return;
  }

  arText = ayah.arabic || "";
  faText = fixFarsiBrackets(ayah.farsi || "");
  surahAndAyah = "سوره " + ayah.surahNameFarsi + "، آیه " + ayah.ayahNumber;

  document.getElementById("arabicVerseText").textContent = arText;
  document.getElementById("verseText").textContent = faText;
  document.getElementById("surahAndAyah").textContent = surahAndAyah;
  document.getElementById("randomMethod").textContent =
    "این آیه، " + toPersianOrdinal(ayah.globalAyah) + " آیه قرآن است.";

  document.getElementById("sourceNote").innerHTML =
    'متن عربی قرآن: <a href="https://tanzil.net" target="_blank" rel="noopener">Tanzil Project</a>';
}
</script>
