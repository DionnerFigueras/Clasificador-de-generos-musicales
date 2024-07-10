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
            /* 
                        // Enviar a la API
                        fetch('/api/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ audio: base64Audio }),
                        })
                            .then((response) => {
                                if (!response.ok) {
                                    // Si hay un error del lado del servidor
                                    throw new Error(`Error HTTP: ${response.status} ${response.statusText}`);
                                }
                                return response.json(); // Parsear la respuesta JSON
                            })
                            .then((data) => {
                                // Éxito: Mostrar alerta con el género
                                alert(`¡Predicción exitosa! Género: ${data.genre}`);
                                predictionDiv.textContent = `Género: ${data.genre}`; // También mostrar en el div
                            })
                            .catch((error) => {
                                // Error: Mostrar alerta con el mensaje de error
                                alert(`Error en la predicción: ${error.message}`);
                                console.error('Error:', error);
                                predictionDiv.textContent = 'Error en la predicción';
                            }); */
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

    return mfccsScaled;
}

// Carga del modelo
let modelo;

async function loadModel() {
  modelo = await tf.loadLayersModel('/modelo/model.json');
  // Modificar la definición del modelo para agregar inputShape
  const newModel = tf.sequential();
  newModel.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [58] }));
  newModel.add(tf.layers.dropout({ rate: 0.3 }));
  newModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  newModel.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  modelo = newModel;
}


async function predecirGenero(audioFile) {
    try {
/*         console.log('Cargando el modelo')
        modelo = await tf.loadLayersModel('/modelo/model.json');
        console.log('Modelo Cargado')
        const mfccsScaled = await procesarAudio(audioFile);

        // Realizar la predicción
        const inputTensor = tf.tensor2d(mfccsScaled, [1, 58]);
        const outputTensor = modelo.predict(inputTensor);
        const probabilities = outputTensor.dataSync();

        // Obtener el índice de la clase con la mayor probabilidad
        const genreIndex = probabilities.indexOf(Math.max(...probabilities));

        // Devolver el género predicho */
        const mfccs = extractMFCCs(audioBuffer);
        const inputTensor = tf.tensor2d(mfccs, [1, 58]);
        const outputTensor = model.predict(inputTensor);
        return outputTensor;
        // return genreIndex;
    } catch (error) {
        alert(`Error al predecir el género: ${error.message}`);
    }
}

// Agregar evento de click al botón de predicción
document.getElementById('uploadButton').addEventListener('click', async () => {
    try {
        const audioInput = document.getElementById('audioInput');
        const audioFile = audioInput.files[0];
        const genreIndex = await predecirGenero(audioFile);
        const genreLabel = le.inverseTransform([genreIndex])[0];
        document.getElementById('prediction').innerHTML = `Predicción del género: ${genreLabel}`;
    } catch (error) {
        alert(`Error al predecir el género: ${error.message}`);
    }
});