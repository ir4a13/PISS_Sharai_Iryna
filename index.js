let video = document.getElementById('camera');
let labels = [];
let xs;
let ys;
let mobilenet;
let model;
let array = Array.from(Array(10), () => 0);
let isPredicting = false;

// Завантажуємо модель
async function loadMobilenet() {
    const mobileNetModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    // Обираємо останній шар
    const layer = mobileNetModel.getLayer('conv_pw_13_relu');
    mobilenet = tf.model({inputs: mobileNetModel.inputs, outputs: layer.output});
}

// ================ Тренування =================
async function train() {
    ys = null;
    // Encode labels as OHE vectors
    encodeLabels(10);
    model = tf.sequential({
        layers: [
            // Отримуємо останній шар з моделі MobileNet
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            // Передаємо результат в Dense шар
            tf.layers.dense({units: 100, activation: 'relu'}),
            // Отримуємо ймовірності класів
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    // Компілюємо модель використовуючи Adam optimizer та крос-ентропію як функцію втрат
    model.compile({optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy'});
    let loss = 0;
    // Тренуємо модель на 10 епох і логуємо втрати
    model.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('Втрати: ' + loss);
            }
        }
    });
}

function encodeLabels(numClasses) {
    for (let i = 0; i < labels.length; i++) {
        const y = tf.tidy(
            () => {
                return tf.oneHot(tf.tensor1d([labels[i]]).toInt(), numClasses)
            });
        if (ys == null) {
            ys = tf.keep(y);
        } else {
            const oldY = ys;
            ys = tf.keep(oldY.concat(y, 0));
            oldY.dispose();
            y.dispose();
        }
    }
}

// Виклик функції з кнопки
function doTraining() {
    train();
    alert("Тренування завершено!")
}

//================================= Створення зразку для тренування ============================================

function addExample(example, label) {
    if (xs === null) {
        xs = tf.keep(example);
    } else {
        const oldX = xs;
        xs = tf.keep(oldX.concat(example, 0));
        oldX.dispose();
    }
    labels.push(label);
}

// Опрацювання кнопок з цифрами для навчання моделі
function handleButton(elem) {
    let label = parseInt(elem.id);
    array[label]++;
    document.getElementById("samples_" + elem.id).innerText = "" + array[label];
    const img = capture();
    addExample(mobilenet.predict(img), label);
}

//================================= Inference process =======================================================

// Процес розпізнавання цифр досить схожий на процес навчання
// безпосередньо - беремо зображення з потоку веб-камери, пропускаємо його через модель MobileNet,
// отримуємо вихід, передаємо його навченій моделі
// і беремо значення з максимальною ймовірністю
async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        console.log((await predictedClass.data())[0]);
        document.getElementById("prediction").innerText = (await predictedClass.data())[0];
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function setPredicting(predicting) {
    isPredicting = predicting;
    predict();
}

function saveModel() {
    model.save('downloads://my_model');
}


// Настройка камери
function adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
        video.width = aspectRatio * video.height;
    } else if (width < height) {
        video.height = video.width / aspectRatio;
    }
}

async function setup() {
    return new Promise((resolve, reject) => {
        // Запитуємо дозвіл на доступ до камери і ставимо розмір відео
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        width: 224,
                        height: 224
                    }
                }).then(stream => {
                video.srcObject = stream;
                video.addEventListener('loadeddata', async () => {
                    adjustVideoSize(video.videoWidth, video.videoHeight);
                    resolve();
                }, false);
            }).catch(error => {
                reject(error);
            });
        } else {
            reject();
        }
    });
}

//Логіка отримання зображення з камери

function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const centerWidth = img.shape[1] / 2;
    const beginHeight = centerHeight - (size / 2);
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function capture() {
    return tf.tidy(() => {
        // Створюємо tf.Tensor з картинки
        const webcamImage = tf.browser.fromPixels(video);
        // Робим реверс картинки (відображення з камери відбувається зліва направо)
        const reversedImage = webcamImage.reverse(1);
        // Ріжемо картинку по центру
        const croppedImage = cropImage(reversedImage);
        const batchedImage = croppedImage.expandDims(0);
        // Нормалізуємо картинку
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

async function init() {
    await setup();
    await loadMobilenet();
}

init();