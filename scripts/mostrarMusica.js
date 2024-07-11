const audioInput = document.getElementById('audioInput');
const uploadButton = document.getElementById('uploadButton');
const predictionDiv = document.getElementById('prediction');
const customButton = document.getElementById('customButton');
const selectedFileP = document.getElementById('selectedFile');

// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', () => {
    customButton.addEventListener('click', () => {
        // Simula un clic en el input de tipo "file"
        audioInput.click();
    });

    // Agrega un event listener para cuando se seleccione un archivo
    audioInput.addEventListener('change', () => {
        const file = audioInput.files[0];
        if (file) {
            // Muestra el nombre del archivo seleccionado
            selectedFileP.textContent = `Archivo seleccionado: ${file.name}`;
        } else {
            selectedFileP.textContent = '';
        }
    });

    // Asignar la función al evento click del botón
    uploadButton.addEventListener('click', () => {
        const file = audioInput.files[0];
        if (!file) {
            alert("Por favor, selecciona un archivo de audio.");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const audioData = e.target.result;

            // Convertir a formato compatible con el modelo (base64 es común)
            const base64Audio = btoa(
                new Uint8Array(audioData).reduce(
                    (data, byte) => data + String.fromCharCode(byte),
                    ''
                )
            );
        };

        reader.readAsArrayBuffer(file);
    });
});

async function procesarAudio(audioFile) {
    // Leer el archivo de audio
    const audioBuffer = await audioFile.arrayBuffer();
    const audioContext = new AudioContext();
    const audioSource = audioContext.createBufferSource();
    audioSource.buffer = await audioContext.decodeAudioData(audioBuffer);

    // Extraer características acústicas (en este caso, Mel-frequency cepstral coefficients)
    const mfccs = [];
    const frameSize = 1024;
    const hopSize = 512;
    const numCoefficients = 58;

    for (let i = 0; i < audioSource.buffer.length; i += hopSize) {
        const frame = audioSource.buffer.slice(i, i + frameSize);
        const mfcc = await librosa.feature.mfcc(frame, {
            sr: audioContext.sampleRate,
            S: null,
            n_mfcc: numCoefficients,
        });
        mfccs.push(mfcc);
    }

    // Convertir las características acústicas en una representación numérica
    const mfccsArray = new Float32Array(mfccs.length * numCoefficients);
    for (let i = 0; i < mfccs.length; i++) {
        mfccsArray.set(mfccs[i], i * numCoefficients);
    }

    // Preprocesar la representación numérica (en este caso, escalar los valores)
    const scaler = new StandardScaler();
    const mfccsScaled = scaler.fitTransform(mfccsArray);

    // Ajustar la forma del array para que sea [1, 58]
    const inputTensor = tf.tensor(mfccsScaled).reshape([1, 58]);

    return inputTensor; // Retornar el tensor con la forma correcta
}


// Carga del modelo
let modelo;

async function cargarModelo() {
    modelo = await tf.loadLayersModel('/modelo/model.json');
}


async function predecirGenero(audioFile) {
    try {
        const mfccs = await procesarAudio(audioFile);
        const inputTensor = tf.tensor2d(mfccs, [, 128]);
        const outputTensor = modelo.predict(inputTensor);
        const prediction = outputTensor.arraySync()[0];
        const genreIndex = prediction.indexOf(Math.max(...prediction));
        return genreIndex;
    } catch (error) {
        alert(`Error al predecir el género: ${error.message}`);
    }
}

const le = {
    labels: ['rock', 'pop', 'jazz', 'classical', 'hiphop', 'electronic', 'folk', 'R&B', 'metal', 'country'],
    inverseTransform: (indices) => {
        return indices.map((index) => this.labels[index]);
    },
    transform: (labels) => {
        return labels.map((label) => this.labels.indexOf(label));
    },
};

// Agregar evento de click al botón de predicción
document.getElementById('uploadButton').addEventListener('click', async () => {
    try {
        const audioInput = document.getElementById('audioInput');
        const audioFile = audioInput.files[0];
        console.log("cargando el modelo");
        await cargarModelo();
        console.log("Modelo cargado");
        const genreIndex = await predecirGenero(audioFile);
        const genreLabel = le.inverseTransform([genreIndex])[0];
        document.getElementById('prediction').innerHTML = `Predicción del género: ${genreLabel}`;
    } catch (error) {
        alert(`Error al predecir el género: ${error.message}`);
    }
});