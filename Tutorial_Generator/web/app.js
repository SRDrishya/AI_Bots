const form = document.querySelector("#tutorial-form");
const topicInput = document.querySelector("#topic");
const goalInput = document.querySelector("#learning-goal");
const generateButton = document.querySelector("#generate-button");
const statusMessage = document.querySelector("#status");

let downloadUrl;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusMessage.classList.remove("error");
  statusMessage.textContent = "Researching sources and creating your tutorial...";
  generateButton.disabled = true;
  generateButton.setAttribute("aria-busy", "true");

  try {
    const response = await fetch("/api/tutorial", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: topicInput.value,
        learning_goal: goalInput.value,
      }),
    });

    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || "The tutorial could not be created.");
    }

    const pdf = await response.blob();
    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl);
    }
    downloadUrl = URL.createObjectURL(pdf);

    const disposition = response.headers.get("Content-Disposition") || "";
    const filename = disposition.match(/filename="([^"]+)"/)?.[1] || "tutorial.pdf";
    const download = document.createElement("a");
    download.href = downloadUrl;
    download.download = filename;
    download.textContent = "Download your PDF tutorial";
    statusMessage.replaceChildren(
      document.createTextNode("Your tutorial is ready. "),
      download,
    );
    download.click();
  } catch (error) {
    statusMessage.classList.add("error");
    statusMessage.textContent = error instanceof Error
      ? error.message
      : "The tutorial could not be created. Please try again.";
  } finally {
    generateButton.disabled = false;
    generateButton.removeAttribute("aria-busy");
  }
});
