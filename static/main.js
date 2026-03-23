const eda = document.querySelector("#eda")
const bvp = document.querySelector("#bvp")
const hr = document.querySelector("#hr")
const temp = document.querySelector("#temp")

const submitBtn = document.querySelector("[type='submit']")

submitBtn.addEventListener("click", async (e)=>{
    e.preventDefault()

    const prediction = await fetchPrediction()

    renderResults(prediction)

    console.log(prediction);
})

async function fetchPrediction(){
    const headers = {
        "Content-Type": "application/json"
    }

    const data = {
        eda: eda.value,
        bvp: bvp.value,
        hr: hr.value,
        temp: temp.value,
    }

    // https://pain-prediction-flask-web.onrender.com
    const resp = await fetch("/predict", {
        method: "POST",
        headers,
        body: JSON.stringify(data)
    })

    const prediction = await resp.json()

    return prediction
}

function renderResults(prediction){
    const painScale = document.querySelectorAll("#result p")[0]
    const painLabel = document.querySelectorAll("#result p")[1]
    const resultTitle = document.querySelector("h4")
    const waitMessage = document.querySelector("h3")

    waitMessage.style.display = "none"
    painLabel.style.display = "block"
    painScale.style.display = "block"
    resultTitle.style.display = "block"
    painScale.textContent = prediction.pain_scale
    painLabel.textContent = prediction.pain_level

}