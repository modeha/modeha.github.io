---
layout: post
title: "Random Quran Ayat Generator"
date: 2026-05-28
categories: [javascript, web, quran]
tags: [html, css, javascript, json, quran, farsi]
---

This small project shows a random Quran Ayah using HTML, CSS, JavaScript, and a local JSON file.

It does not use any online Quran API. The Arabic text and Farsi translation are loaded from a local JSON file generated from the PDF translation by استاد حسین انصاریان.

The random selection uses a weighted method based on factors related to the number 19.

<div class="quran-container">
  <div class="quoteBox">
    <h1>Random Quran Ayat Generator</h1>

    <div id="content">
      <div id="arabicVerseText">Loading Quran data...</div>
      <div id="verseText" class="mediumSize"></div>
      <div id="surahAndAyah" class="mediumSize"></div>
      <div id="randomMethod"></div>
      <div id="errorMessage"></div>
    </div>

    <input id="shuffle" type="button" value="New Ayat">
    <input id="tweet" type="button" value="Tweet">
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
  padding: 60px 50px;
  background-color: #fff;
  border: 1px solid #ddd;
}

.quoteBox h1 {
  text-align: center;
  font-size: 250%;
  margin-bottom: 40px;
}

#content {
  margin-bottom: 40px;
}

#arabicVerseText {
  font-size: 220%;
  direction: rtl;
  margin: 25px 0;
  line-height: 1.8;
}

#verseText {
  direction: rtl;
  text-align: right;
}

.mediumSize {
  font-size: 150%;
  margin: 20px 0;
  line-height: 1.8;
}

#randomMethod {
  font-size: 110%;
  color: #555;
  margin-top: 20px;
}

#errorMessage {
  font-size: 120%;
  color: #b00020;
  margin-top: 20px;
}

input[type=button] {
  cursor: pointer;
  padding: 12px 28px;
  margin: 10px;
  border: none;
  border-radius: 5px;
  background-color: #1192d3;
  color: white;
  font-size: 18px;
}

input[type=button]:hover {
  background-color: #0d75aa;
}

@media (max-width: 650px) {
  .quran-container {
    width: 92%;
  }

  .quoteBox {
    padding: 30px 20px;
  }

  .quoteBox h1 {
    font-size: 200%;
  }

  #arabicVerseText {
    font-size: 180%;
  }

  .mediumSize {
    font-size: 130%;
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
      alert("Please wait until the Ayah loads.");
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
      "The local Quran JSON file could not be loaded. Make sure quran_fa_ansarian.json is inside assets/data.";

    console.error(error);
  }
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
    document.getElementById("arabicVerseText").textContent = "Loading Quran data...";
    return;
  }

  arText = ayah.arabic || "";
  faText = ayah.farsi || "";
  surahAndAyah = "سوره " + ayah.surahNameFarsi + "، آیه " + ayah.ayahNumber;

  document.getElementById("arabicVerseText").textContent = arText;
  document.getElementById("verseText").textContent = faText;
  document.getElementById("surahAndAyah").textContent = surahAndAyah;
  document.getElementById("randomMethod").textContent =
    "Selected using advanced 19-based weighted random method. Global Ayah: " + ayah.globalAyah;
}
</script>
