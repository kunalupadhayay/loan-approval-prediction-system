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

    const originalLabel = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = "Reading the ledger…";

    const payload = Object.fromEntries(new FormData(form).entries());

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Request failed");

      const data = await response.json();
      renderResult(data);
    } catch (err) {
      emptyState.hidden = true;
      resultState.hidden = true;
      errorState.hidden = false;
      formHint.textContent = "Something went wrong. Try again.";
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
    confidenceFill.classList.toggle("confidence-fill-reject", !approved);

    const pct = approved ? data.approve_probability : data.approve_probability;
    confidenceValue.textContent = data.approve_probability;
    // animate the bar
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${data.approve_probability}%`;
    });

    reasonsList.innerHTML = "";
    data.reasons.forEach((reason) => {
      const li = document.createElement("li");
      li.className = `reason-${reason.type}`;
      li.textContent = reason.text;
      reasonsList.appendChild(li);
    });

    resultState.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
