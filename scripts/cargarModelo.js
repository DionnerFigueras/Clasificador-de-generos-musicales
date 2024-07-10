
let modelo = null;
let le = null; // Para decodificar las predicciones

async function cargarModelo() {
    console.log("Cargando modelo...");
    modelo = await tf.loadLayersModel('/modelo/model.json');
    console.log("Modelo cargado!");

    // Define la forma de entrada aquí (no es necesario modificarla dinámicamente)
    const inputShape = [58];
    modelo.layers[0].inputShape = [null, ...inputShape];

    // Ejemplo de cómo cargar el LabelEncoder (necesitarás adaptarlo)
    const respuestaLe = await fetch('/modelo/label_encoder.json'); // Ajusta la ruta
    le = await respuestaLe.json();
    console.log("Label Encoder cargado:", le);
}

async function predecirGenero(audioData) {
    if (!modelo) {
        console.error("El modelo no está cargado.");
        return;
    }

    try {
        // 1. Decodificar el audio (ya lo haces con FileReader)
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(audioData);

        // 2. Extraer características MFCC
        const mfccs = extraerMFCCs(audioBuffer, audioContext.sampleRate);
        console.log("MFCCs:", mfccs);

        // 3. Convertir a tensor de TensorFlow.js
        const tensor = tf.tensor(mfccs).reshape([1, mfccs.length]); // Ajustar la forma

        // 4. Realizar la predicción
        const prediccion = modelo.predict(tensor);
        console.log("Predicción del modelo:", prediccion);

        // 5. Obtener la clase con la probabilidad más alta
        const indiceClase = Array.from(prediccion.dataSync()).indexOf(Math.max(...prediccion.dataSync()));
        console.log("Índice de clase:", indiceClase);

        // 6. Decodificar la predicción usando el LabelEncoder
        const generoPredicho = le[indiceClase]; // Ajusta según la estructura de tu LabelEncoder
        console.log("Género predicho:", generoPredicho);

        // 7. Mostrar la predicción al usuario
        predictionDiv.textContent = `Género: ${generoPredicho}`;
    } catch (error) {
        console.error("Error al predecir el género:", error);
        predictionDiv.textContent = 'Error en la predicción';
    }
}

// Función para extraer MFCCs (debes implementarla o usar una librería)
function extraerMFCCs(audioBuffer, sampleRate) {
    // ... Implementa la extracción de MFCCs usando una librería como librosa.js ...
    // Asegúrate de que la salida tenga la forma [número_de_mfccs]
}

// ... (Resto del código del script) ...

uploadButton.addEventListener('click', () => {
    // ... (Código para cargar el archivo de audio) ...

    reader.onload = (e) => {
        const audioData = e.target.result;
        predecirGenero(audioData);
    };

    reader.readAsArrayBuffer(file);
});

// Cargar el modelo al inicio
cargarModelo();
