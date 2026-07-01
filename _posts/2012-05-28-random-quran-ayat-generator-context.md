---
layout: post
title: "آیه تصادفی قرآن"
date: 2026-05-28
categories: [javascript, web, quran]
tags: [html, css, javascript, json, quran, farsi]
---

<div class="intro-text">
  <p>این پروژه یک آیه تصادفی قرآن را همراه با ترجمه فارسی نشان می‌دهد.</p>
  <p>انتخاب آیه با یک روش وزن‌دهی شده بر اساس عدد ۱۹ انجام می‌شود.</p>
</div>


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

    <div class="contextControls">
      <label for="contextCount">تعداد آیات قبل و بعد:</label>
      <input id="contextCount" type="number" min="1" max="50" value="10">
      <input id="showContext" type="button" value="نمایش آیات قبل و بعد">
    </div>

    <div id="contextAyat"></div>
  </div>
</div>

<style>
@font-face {
  font-family: "UthmanicHafs";
  src: url('{{ "/assets/fonts/UthmanicHafs1%20Ver16.ttf" | relative_url }}') format("truetype");
  font-weight: normal;
  font-style: normal;
}
.intro-text {
  text-align: center;
  direction: rtl;
  max-width: 700px;
  margin: 30px auto;
}
.quran-container {
  text-align: center;
  border-radius: 5px;
  position: relative;
  margin: 15px auto;
  width: 80%;
  max-width: 900px;
  background-color: #fff;
}

.quoteBox {
  border-radius: 5px;
  position: relative;
  margin: 15px auto;
  padding: 40px 45px;
  background-color: #fff;
  border: 1px solid #ddd;
}

.quran-title {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  font-size: 120%;
  font-weight: normal;
  direction: rtl;
  text-align: center;
  margin-bottom: 25px;
  color: #333;
}

#content {
  margin-bottom: 30px;
}

#arabicVerseText {
  font-family: "UthmanicHafs", "Amiri", "Scheherazade New", "Traditional Arabic", serif;
  font-size: 220%;
  direction: rtl;
  text-align: center;
  margin: 20px 0;
  line-height: 2.1;
}

#verseText {
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  direction: rtl;
  text-align: center;
  unicode-bidi: plaintext;
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


.contextControls {
  margin-top: 15px;
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
  direction: rtl;
  text-align: center;
}

#contextCount {
  width: 70px;
  padding: 6px;
  margin: 5px;
  text-align: center;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-family: Tahoma, Arial, sans-serif;
}

#contextAyat {
  margin-top: 25px;
  direction: rtl;
  text-align: right;
  font-family: "UthmanicHafs", Tahoma, Arial, sans-serif;
}

.contextItem {
  border-top: 1px solid #ddd;
  padding: 15px 0;
}

.contextHeader {
  font-size: 105%;
  color: #333;
  text-align: center;
  margin-bottom: 8px;
}

.contextArabic {
  font-size: 145%;
  line-height: 2;
  text-align: center;
  direction: rtl;
  margin: 8px 0;
}

.contextFarsi {
  font-size: 105%;
  line-height: 1.8;
  text-align: right;
  direction: rtl;
  unicode-bidi: plaintext;
  margin: 8px 0;
}

.contextCurrent {
  background-color: #f7fbff;
  padding: 15px;
  border-radius: 6px;
  border: 1px solid #d7ecff;
}

@media (max-width: 650px) {
  .quran-container {
    width: 92%;
  }

  .quoteBox {
    padding: 25px 18px;
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

  .contextArabic {
    font-size: 125%;
  }

  .contextFarsi {
    font-size: 100%;
  }

  .contextControls input[type=button] {
    display: block;
    margin: 10px auto;
  }
}
</style>

<script>
let quranAyat = [];
let arText = "";
let faText = "";
let surahAndAyah = "";
let selectedAyah = null;

document.addEventListener("DOMContentLoaded", async function () {
  await loadQuranData();

  document.getElementById("shuffle").addEventListener("click", showRandomAyah);
  document.getElementById("showContext").addEventListener("click", showAyahContext);
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
    document.getElementById("contextAyat").innerHTML = "";
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
  const special = {
    1: "اولین",
    2: "دومین",
    3: "سومین",
    4: "چهارمین",
    5: "پنجمین",
    6: "ششمین",
    7: "هفتمین",
    8: "هشتمین",
    9: "نهمین",
    10: "دهمین",
    20: "بیستمین",
    30: "سی‌امین",
    40: "چهلمین",
    50: "پنجاهمین",
    60: "شصتمین",
    70: "هفتادمین",
    80: "هشتادمین",
    90: "نودمین",
    100: "صدمین",
    200: "دویستمین",
    300: "سیصدمین",
    400: "چهارصدمین",
    500: "پانصدمین",
    600: "ششصدمین",
    700: "هفتصدمین",
    800: "هشتصدمین",
    900: "نهصدمین"
  };

  if (special[n]) {
    return special[n];
  }

  if (n < 100) {
    const t = Math.floor(n / 10) * 10;
    const r = n % 10;
    return numberToPersianWords(t) + " و " + toPersianOrdinal(r);
  }

  if (n < 1000) {
    const h = Math.floor(n / 100) * 100;
    const r = n % 100;
    return numberToPersianWords(h) + " و " + toPersianOrdinal(r);
  }

  const th = Math.floor(n / 1000);
  const r = n % 1000;

  if (r === 0) {
    return numberToPersianWords(th) + " هزارمین";
  }

  return numberToPersianWords(th) + " هزار و " + toPersianOrdinal(r);
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


function showAyahContext() {
  if (!selectedAyah || !quranAyat.length) {
    return;
  }

  let count = Number(document.getElementById("contextCount").value) || 10;

  if (count < 1) {
    count = 1;
  }

  if (count > 50) {
    count = 50;
  }

  document.getElementById("contextCount").value = count;

  const start = Math.max(1, selectedAyah.globalAyah - count);
  const end = Math.min(quranAyat.length, selectedAyah.globalAyah + count);

  const selectedRange = quranAyat.filter(function (ayah) {
    return ayah.globalAyah >= start && ayah.globalAyah <= end;
  });

  const html = selectedRange.map(function (ayah) {
    const isCurrent = ayah.globalAyah === selectedAyah.globalAyah;

    return `
      <div class="contextItem ${isCurrent ? "contextCurrent" : ""}">
        <div class="contextHeader">
          <strong>سوره ${ayah.surahNameFarsi}، آیه ${ayah.ayahNumber}</strong>
        </div>
        <div class="contextArabic">${ayah.arabic || ""}</div>
        <div class="contextFarsi">${fixFarsiBrackets(ayah.farsi || "")}</div>
      </div>
    `;
  }).join("");

  document.getElementById("contextAyat").innerHTML = html;
}

function showRandomAyah() {
  document.getElementById("errorMessage").textContent = "";
  document.getElementById("contextAyat").innerHTML = "";

  const ayah = getAdvanced19RandomAyah();

  if (!ayah) {
    document.getElementById("arabicVerseText").textContent = "در حال بارگذاری...";
    return;
  }

  selectedAyah = ayah;

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
