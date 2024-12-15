---
layout: post
title: "Correlation and Angle Relationship"
date: 2024-12-14
categories: [math, data-science]
tags: [correlation, angle, math]
---

# Correlation and Angle Relationship

The angle between two vectors (features) depends on their correlation coefficient, as it is directly related to the cosine of the angle between them:

\[
r = \cos(\theta)
\]

where \( r \) is the correlation coefficient (\(-1 \leq r \leq 1\)), and \( \theta \) is the angle between the two vectors.

---

## Relationship Between Correlation and Angle

### High Positive Correlation (\( r \approx +1 \)):
- When \( r \to 1 \), \( \cos(\theta) \to 1 \), meaning \( \theta \to 0^\circ \).
- The vectors are nearly aligned in the same direction.
- **Example:** If \( r = 0.9 \), the angle is small:
  \[
  \theta = \cos^{-1}(0.9) \approx 25.84^\circ
  \]

### High Negative Correlation (\( r \approx -1 \)):
- When \( r \to -1 \), \( \cos(\theta) \to -1 \), meaning \( \theta \to 180^\circ \).
- The vectors are nearly aligned in opposite directions.
- **Example:** If \( r = -0.9 \), the angle is:
  \[
  \theta = \cos^{-1}(-0.9) \approx 154.16^\circ
  \]

### Small Correlation (\( r \) is small):
- When \( r \) is small (e.g., \( r \approx 0.1 \) or \( r \approx -0.1 \)), \( \cos(\theta) \) is close to \( 0 \), meaning \( \theta \) is close to \( 90^\circ \).
- The vectors are nearly orthogonal.
- **Example:** If \( r = 0.1 \), the angle is:
  \[
  \theta = \cos^{-1}(0.1) \approx 84.26^\circ
  \]

### Zero Correlation (\( r = 0 \)):
- When \( r = 0 \), \( \cos(\theta) = 0 \), meaning \( \theta = 90^\circ \).
- The vectors are orthogonal or perpendicular.

---

## General Cases

| Correlation Coefficient (\( r \)) | Angle (\( \theta \))       | Relationship                          |
|-----------------------------------|----------------------------|---------------------------------------|
| \( r = 1 \)                       | \( 0^\circ \)              | Perfect positive alignment            |
| \( 0 < r < 1 \)                   | \( 0^\circ < \theta < 90^\circ \) | Small acute angle (positive correlation) |
| \( r = 0 \)                       | \( 90^\circ \)             | Orthogonal (no linear relationship)   |
| \( -1 < r < 0 \)                  | \( 90^\circ < \theta < 180^\circ \) | Obtuse angle (negative correlation)   |
| \( r = -1 \)                      | \( 180^\circ \)            | Perfect negative alignment            |

---

## Examples with Calculations

#### High Positive Correlation (\( r = 0.8 \)):
\[
\theta = \cos^{-1}(0.8) \approx 36.87^\circ
\]

#### Low Positive Correlation (\( r = 0.2 \)):
\[
\theta = \cos^{-1}(0.2) \approx 78.46^\circ
\]

#### High Negative Correlation (\( r = -0.8 \)):
\[
\theta = \cos^{-1}(-0.8) \approx 143.13^\circ
\]

#### Small Negative Correlation (\( r = -0.2 \)):
\[
\theta = \cos^{-1}(-0.2) \approx 101.54^\circ
\]

---

## Conclusion

**High correlation (\( r \to 1 \))** results in small angles (\( \theta \to 0^\circ \) or \( \theta \to 180^\circ \)).
