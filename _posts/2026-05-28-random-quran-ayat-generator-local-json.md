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
      <div id="errorMessage"></div>
    </div>

    <input id="shuffle" type="button" value="آیه جدید">
    <input id="tweet" type="button" value="اشتراک‌گذاری">
  </div>
</div>

<style>
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
  font-family: Tahoma, Arial, sans-serif;
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
  font-family: "Amiri", "Scheherazade New", "Traditional Arabic", serif;
  font-size: 240%;
  direction: rtl;
  text-align: center;
  margin: 20px 0;
  line-height: 2.1;
}

#verseText {
  direction: rtl;
  text-align: center;
  font-family: Tahoma, Arial, sans-serif;
  unicode-bidi: plaintext;
}

.mediumSize {
  font-size: 140%;
  margin: 16px 0;
  line-height: 1.8;
}

#randomMethod {
  font-size: 85%;
  color: #666;
  margin-top: 18px;
  direction: ltr;
}

#errorMessage {
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
  font-family: Tahoma, Arial, sans-serif;
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
  }

  .quran-title {
    font-size: 110%;
  }

  #arabicVerseText {
    font-size: 190%;
  }

  .mediumSize {
    font-size: 120%;
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

  document.getElementById("tweet").addEventListener("click", function () {
    if (!faText || !surahAndAyah) {
      alert("لطفاً صبر کنید تا آیه بارگذاری شود.");
      return;
    }

    const tweetText = arText + "\n\n" + faText + "\n" + surahAndAyah;
    const tweetLink = "https://twitter.com/intent/tweet?text=" + encodeURIComponent(tweetText);
    window.open(tweetLink, "_blank");
  });
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
    "روش انتخاب: وزن‌دهی بر اساس عدد ۱۹ | شماره جهانی آیه: " + ayah.globalAyah;
}
</script>
