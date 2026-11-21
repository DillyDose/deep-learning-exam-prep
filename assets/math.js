// Renders \( inline \) and \[ display \] / $$ display $$ math with KaTeX (self-hosted in assets/katex).
document.addEventListener("DOMContentLoaded", () => {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false,
    // colour code (study-card lessons): random noise = blue, what the network outputs/learns = orange
    macros: { "\\rnd": "\\textcolor{##1d6fb8}{#1}", "\\net": "\\textcolor{##c25e00}{#1}" },
  });
});
