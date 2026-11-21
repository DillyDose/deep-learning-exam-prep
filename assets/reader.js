// Reader helpers: reading-progress bar, click-to-zoom for slide images, open folded sections when printing.
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.createElement("div");
  bar.id = "readbar";
  document.body.appendChild(bar);
  const onScroll = () => {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = (h > 0 ? (scrollY / h) * 100 : 0) + "%";
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const zoom = document.createElement("div");
  zoom.id = "zoom";
  zoom.innerHTML = "<img alt=\"\">";
  document.body.appendChild(zoom);
  const big = zoom.querySelector("img");
  document.querySelectorAll("figure.thumb img, .gallery img").forEach((img) => {
    img.addEventListener("click", () => {
      big.src = img.src;
      big.alt = img.alt;
      zoom.classList.add("on");
    });
  });
  zoom.addEventListener("click", () => zoom.classList.remove("on"));
  addEventListener("keydown", (e) => { if (e.key === "Escape") zoom.classList.remove("on"); });

  const openTo = (hash) => {
    const t = hash && document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!t) return;
    for (let el = t.parentElement; el; el = el.parentElement) if (el.tagName === "DETAILS") el.open = true;
    t.scrollIntoView({ block: "center" });
  };
  document.querySelectorAll('a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
    const h = a.getAttribute("href");
    if (h.length > 1 && document.getElementById(decodeURIComponent(h.slice(1)))) { e.preventDefault(); history.replaceState(null, "", h); openTo(h); }
  }));
  if (location.hash) openTo(location.hash);

  addEventListener("beforeprint", () => document.querySelectorAll("details.fold").forEach((d) => (d.open = true)));
});
