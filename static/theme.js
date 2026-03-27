// Global theme controller for all pages (no dependencies).
(function () {
  const STORAGE_KEY = "theme_preference"; // "system" | "light" | "dark"

  function readPref() {
    const v = (localStorage.getItem(STORAGE_KEY) || "system").toLowerCase();
    return v === "light" || v === "dark" || v === "system" ? v : "system";
  }

  function setPref(pref) {
    localStorage.setItem(STORAGE_KEY, pref);
  }

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme() {
    const pref = readPref();
    const resolved = pref === "system" ? systemTheme() : pref;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themePref = pref;
    window.dispatchEvent(new CustomEvent("themechange", { detail: { pref, resolved } }));
    return { pref, resolved };
  }

  function setThemePreference(pref) {
    setPref(pref);
    return applyTheme();
  }

  // Expose minimal API
  window.Theme = {
    getPreference: readPref,
    apply: applyTheme,
    setPreference: setThemePreference,
  };

  // Apply immediately
  applyTheme();

  // Live update if OS theme changes while in "system"
  if (window.matchMedia) {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      if (readPref() === "system") applyTheme();
    };
    if (typeof mql.addEventListener === "function") mql.addEventListener("change", handler);
    else if (typeof mql.addListener === "function") mql.addListener(handler);
  }

  // Sync across tabs
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEY) applyTheme();
  });
})();

