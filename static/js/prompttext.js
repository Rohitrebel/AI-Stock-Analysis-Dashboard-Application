document.addEventListener("DOMContentLoaded", () => {
  let mainEl = document.getElementById("main-container");
  let urlform = document.getElementById("url-id");
  let csvform = document.getElementById("csv-id");
  let textform = document.getElementById("text-id");

  function handleForm(form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      const formData = new FormData(this);

      const submitBtn = event.submitter; // the button clicked
      if (submitBtn && submitBtn.name) {
        formData.append(submitBtn.name, submitBtn.value);
      }

      // Show spinner while processing
      mainEl.innerHTML = `<div id="loading-spinner" >⏳ Processing...</div>`;

      fetch("/news-research-tool", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.status === "done") {
            mainEl.innerHTML = `
              <div class="prompt-section">
                <input type="text" id="prompt-text" placeholder="Enter your query" class="prompt-input"/>
                <button id="submit-btn-mid" class="submit-btn-tool">Ask</button>
                <div id="loading-answer" style="display:none;">⚡ Thinking...</div>
                <div id="results-container"></div>
              </div>
            `;

            document
              .getElementById("submit-btn-mid")
              .addEventListener("click", afterContinue);
          } else {
            mainEl.innerHTML = `<p style="color:red;">❌ Error: ${data.message}</p>`;
          }
        })
        .catch(() => {
          mainEl.innerHTML = `<p style="color:red;">⚠️ Failed to connect to server.</p>`;
        });
    });
  }

  handleForm(urlform);
  handleForm(csvform);
  handleForm(textform);
});

function afterContinue() {
  const value = document.getElementById("prompt-text").value;
  const resultsDiv = document.getElementById("results-container");
  const loadingDiv = document.getElementById("loading-answer");

  resultsDiv.textContent = "";
  loadingDiv.style.display = "block"; // show "Thinking..." spinner

  fetch("/continueprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userPrompt: value }),
  })
    .then((response) => response.json())
    .then((data) => {
      loadingDiv.style.display = "none";
      resultsDiv.innerHTML = `<b>Answer:</b> ${data.answer}`;
    })
    .catch(() => {
      loadingDiv.style.display = "none";
      resultsDiv.innerHTML = `<p style="color:red;">⚠️ Error getting response</p>`;
    });
}
