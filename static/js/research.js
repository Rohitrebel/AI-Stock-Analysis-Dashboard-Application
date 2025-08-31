document.addEventListener("DOMContentLoaded", () => {
  let selecturlEl = document.getElementById("select-box");
  let textinpEl = document.getElementById("text-id");
  let urlinpEl = document.getElementById("url-id");
  let csvinpEl = document.getElementById("csv-id");
  selecturlEl.addEventListener("change", (event) => {
    let dataType = event.target.value;

    if (dataType === "url") {
      if (!textinpEl.classList.contains("d-text-hide")) {
        textinpEl.classList.add("d-text-hide");
      }
      if (!csvinpEl.classList.contains("d-csv-hide")) {
        csvinpEl.classList.add("d-csv-hide");
      }
      if (urlinpEl.classList.contains("d-url-hide")) {
        urlinpEl.classList.remove("d-url-hide");
      }
    }
    if (dataType === "text") {
      if (textinpEl.classList.contains("d-text-hide")) {
        textinpEl.classList.remove("d-text-hide");
      }
      if (!csvinpEl.classList.contains("d-csv-hide")) {
        csvinpEl.classList.add("d-csv-hide");
      }
      if (!urlinpEl.classList.contains("d-url-hide")) {
        urlinpEl.classList.add("d-url-hide");
      }
    }
    if (dataType === "csv") {
      if (!textinpEl.classList.contains("d-text-hide")) {
        textinpEl.classList.add("d-text-hide");
      }
      if (csvinpEl.classList.contains("d-csv-hide")) {
        csvinpEl.classList.remove("d-csv-hide");
      }
      if (!urlinpEl.classList.contains("d-url-hide")) {
        urlinpEl.classList.add("d-url-hide");
      }
    }
  });
});
