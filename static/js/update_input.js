document.addEventListener("DOMContentLoaded", () => {
  document.addEventListener("click", (event) => {
    const plusBtn = event.target.closest(".plus-container");
    const minusBtn = event.target.closest(".minus-container");

    if (plusBtn || minusBtn) {
      const widget = (plusBtn || minusBtn).closest(".inc-dec-container");
      const inputEl = widget.querySelector(".returns-input");

      if (inputEl.type === "date") {
        // Handle date increment/decrement
        let currentDate = new Date(inputEl.value);
        if (!isNaN(currentDate)) {
          currentDate.setDate(currentDate.getDate() + (plusBtn ? 1 : -1));
          inputEl.value = currentDate.toISOString().split("T")[0];
        }
      } else {
        // Handle numeric increment/decrement
        let currentValue = parseInt(inputEl.value, 10);
        if (isNaN(currentValue)) currentValue = 0;

        if (plusBtn) {
          inputEl.value = currentValue + 1;
        } else if (minusBtn && currentValue > 0) {
          inputEl.value = currentValue - 1;
        }
      }
    }
  });
});
