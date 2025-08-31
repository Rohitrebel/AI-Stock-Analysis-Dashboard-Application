document.addEventListener("DOMContentLoaded", () => {
  let tickerSubmitEl = document.getElementById("ticker-submit-id");
  let downloadRequestEl = document.getElementById("downloadbtnContainer");

  function handleform(form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = new FormData(form);

      const submitbtn = event.submitter;
      if (submitbtn && submitbtn.name) {
        formData.append(submitbtn.name, submitbtn.value);
      }

      downloadRequestEl.innerHTML = `<div id="loading-spinner">⏳ Fetching...</div>`;

      if (submitbtn.name === "submit-request") {
        fetch("/performance-summary", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.status === "done") {
              // ✅ inject form that triggers download
              downloadRequestEl.innerHTML = `
                <form action="/performance-summary" method="POST">
                  <input type="hidden" name="download-request" value="1" />
                  <button type="submit" class="submit-btn-tool download-csv-btn">Download CSV</button>
                </form>
              `;
            }
          });
      }
    });
  }

  handleform(tickerSubmitEl);
});
