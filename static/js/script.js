document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loan-form");
  if (!form) return;

  const emptyState = document.getElementById("decision-empty");
  const resultState = document.getElementById("decision-result");
  const errorState = document.getElementById("decision-error");
  const stamp = document.getElementById("stamp");
  const stampText = document.getElementById("stamp-text");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidenceValue = document.getElementById("confidence-value");
  const reasonsList = document.getElementById("reasons-list");
  const submitBtn = form.querySelector(".btn-primary");
  const formHint = document.getElementById("form-hint");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Check browser-side validation
    if (!form.reportValidity()) return;

    const originalLabel = submitBtn.textContent;

    submitBtn.disabled = true;
    submitBtn.textContent = "Reading the ledger…";

    emptyState.hidden = true;
    resultState.hidden = true;
    errorState.hidden = true;
    formHint.textContent = "Processing your application…";

    const payload = Object.fromEntries(new FormData(form).entries());

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(payload)
      });

      // Read the backend response, including errors
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status})`);
      }

      if (!data.prediction) {
        throw new Error("The server response does not contain a prediction.");
      }

      renderResult(data);

      formHint.textContent = "Assessment completed successfully.";

    } catch (err) {
      console.error("Loan prediction error:", err);

      emptyState.hidden = true;
      resultState.hidden = true;
      errorState.hidden = false;

      // Show the actual error returned by Flask
      const errorParagraph = errorState.querySelector("p");

      if (errorParagraph) {
        errorParagraph.textContent = err.message;
      }

      formHint.textContent = "Assessment failed. Check the error message.";

    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = originalLabel;
    }
  });

  function renderResult(data) {
    emptyState.hidden = true;
    errorState.hidden = true;
    resultState.hidden = false;

    const approved = data.prediction === "Approved";

    stampText.textContent = data.prediction;

    stamp.classList.toggle("stamp-reject", !approved);
    confidenceFill.classList.toggle(
      "confidence-fill-reject",
      !approved
    );

    const confidence = Number(
      approved
        ? data.approve_probability
        : data.reject_probability
    );

    confidenceValue.textContent = confidence.toFixed(1);

    requestAnimationFrame(() => {
      confidenceFill.style.width =
        `${Math.max(0, Math.min(100, confidence))}%`;
    });

    reasonsList.replaceChildren();

    (data.reasons || []).forEach((reason) => {
      const li = document.createElement("li");

      li.className = `reason-${reason.type}`;
      li.textContent = reason.text;

      reasonsList.appendChild(li);
    });

    resultState.scrollIntoView({
      behavior: "smooth",
      block: "nearest"
    });
  }
});