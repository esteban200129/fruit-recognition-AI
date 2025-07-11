document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("image-upload");
    const file = fileInput.files[0];
    const allowedTypes = ["image/jpeg", "image/png"];

    if (!file) {
        alert("Please upload an image file!");
        return;
    }

    if (!allowedTypes.includes(file.type)) {
        alert("Please upload a valid image file (JPEG/PNG)!");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    // 預覽圖片
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById("preview-img");
        previewImg.src = e.target.result;
        previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to get prediction");
        }

        const result = await response.json();
        if (result.error) {
            alert(`Error: ${result.error}`);
            return;
        }

        // 顯示預測結果
        document.getElementById("prediction").textContent = `Prediction: ${result.prediction}`;

        // 顯示營養資訊
        const nutrientsList = document.getElementById("nutrients-list");
        nutrientsList.innerHTML = ""; // 清空舊數據
        if (result.nutrients.message) {
            nutrientsList.innerHTML = `<li>${result.nutrients.message}</li>`;
        } else {
            Object.entries(result.nutrients).forEach(([key, value]) => {
                const li = document.createElement("li");
                li.textContent = `${key}: ${value}`;
                nutrientsList.appendChild(li);
            });
        }

    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while predicting the image.");
    }
});