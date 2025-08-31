document.addEventListener("DOMContentLoaded", () => {
  let navEl = document.getElementById("nav-bar");
  let navItemsEl = document.getElementById("nav-items");
  let crossbarEl = document.getElementById("cross-bars-el");
  navEl.addEventListener("click", () => {
    crossbarEl.classList.add("cross-bars-sm");
    navItemsEl.classList.add("nav-show");
    navEl.classList.remove("bars-sm");
  });
  crossbarEl.addEventListener("click", () => {
    navEl.classList.add("bars-sm");
    navItemsEl.classList.remove("nav-show");
    crossbarEl.classList.remove("cross-bars-sm");
  });
});
