(() => {
  const revealNodes = Array.from(document.querySelectorAll(".reveal"));
  const buttons = Array.from(document.querySelectorAll(".interactive-btn"));
  const iframe = document.querySelector("#interactive-frame");
  const interactiveTitle = document.querySelector("#interactive-title");
  const interactiveSummary = document.querySelector("#interactive-summary");

  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { rootMargin: "0px 0px -12% 0px", threshold: 0.15 }
  );

  for (const node of revealNodes) observer.observe(node);

  for (const btn of buttons) {
    btn.addEventListener("click", () => {
      const target = btn.dataset.src;
      if (!target || !iframe) return;
      for (const b of buttons) b.classList.remove("active");
      btn.classList.add("active");
      iframe.src = target;
      if (interactiveTitle && btn.dataset.title) interactiveTitle.textContent = btn.dataset.title;
      if (interactiveSummary && btn.dataset.summary) interactiveSummary.textContent = btn.dataset.summary;
    });
  }

  const videos = Array.from(document.querySelectorAll("video"));
  for (const video of videos) {
    video.addEventListener(
      "error",
      () => {
        const fallback = document.createElement("p");
        fallback.className = "note";
        fallback.textContent = "Video asset missing. Add the expected MP4 file to the site root or update the source path in index.html.";
        const host = video.closest(".media-copy") || video.parentElement;
        if (host) host.appendChild(fallback);
      },
      { once: true }
    );
  }
})();
