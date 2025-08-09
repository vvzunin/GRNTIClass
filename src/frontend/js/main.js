function hideDownloadButton() {
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.classList.add('hidden');
    setTimeout(() => {
        downloadBtn.style.display = 'none';
    }, 700); // Совпадает с длительностью анимации
}

function showDownloadButton() {
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'block';
    setTimeout(() => {
        downloadBtn.classList.remove('hidden');
    }, 10);
}
  

document.addEventListener('DOMContentLoaded', function() {

    document.getElementById('level1').addEventListener('change', hideDownloadButton);
    document.getElementById('level2').addEventListener('change', hideDownloadButton);
    document.getElementById('level3').addEventListener('change', hideDownloadButton);
    document.getElementById('decoding').addEventListener('change', hideDownloadButton);
    document.getElementById('probabilitySlider').addEventListener('input', hideDownloadButton);

    // Делаем функции доступными для других частей кода
    window.hideDownloadButton = hideDownloadButton;
    window.showDownloadButton = showDownloadButton;

    // Обработка изменения положения ползунка
    const slider = document.getElementById('probabilitySlider');
    const sliderValue = document.getElementById('sliderValue');
    
    slider.addEventListener('input', function() {
        const value = this.value / 100;
        sliderValue.textContent = value.toFixed(2);
    });

    const classifyBtn = document.getElementById('classifyBtn');

    function showErrorNotification(message) {
        const notification = document.getElementById('errorNotification');
        const messageElement = document.getElementById('errorMessage');
        
        messageElement.textContent = message;
        notification.classList.add('show');
        
        setTimeout(hideErrorNotification, 5000);
    }
    
    function hideErrorNotification() {
        document.getElementById('errorNotification').classList.remove('show');
    }

    function updateProgress(progress, message, currentFile, totalFiles) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = message;
    }
    
    // Обработка кнопки подтверждения
    document.getElementById('confirmBtn').addEventListener('click', async function() {
        document.getElementById('confirmationModal').style.display = 'none';
        const loadingModal = document.getElementById('loadingModal');
        loadingModal.style.display = 'flex';
        
        try {
            const params = {
                level1: document.getElementById('level1').checked,
                level2: document.getElementById('level2').checked,
                level3: document.getElementById('level3').checked,
                decoding: document.getElementById('decoding').checked,
                threshold: slider.value / 100
            };
            
            const files = fileHandler.getFiles();
            
            // Сбрасываем прогресс перед началом
            updateProgress(0, "Подготовка к обработке...", 0, files.length);
            
            console.log('Начинаем классификацию...');
            const classificationResults = await API.classify(
                files, 
                params, 
                (progress, message, current, total) => {
                    updateProgress(progress, message, current, total);
                }
            );
            
            console.log('Результаты получены:', classificationResults);
            displayResults(classificationResults, params.decoding);
            loadingModal.style.display = 'none';
            
        } catch (error) {
            console.error('Ошибка классификации:', error);
            showErrorNotification(error.message);
            loadingModal.style.display = 'none';
        }
    });

    classifyBtn.addEventListener('click', function() {
        
        const level1 = document.getElementById('level1').checked;
        const level2 = document.getElementById('level2').checked;
        const level3 = document.getElementById('level3').checked;
        const decoding = document.getElementById('decoding').checked;
        const threshold = slider.value / 100;

        // Формируем текст подтверждения
        let details = '<p><strong>Выбранные файлы:</strong></p><ul>';
        const files = fileHandler.getFiles();

        files.forEach(file => {
            details += `<li>${file.name} (${fileHandler.formatFileSize(file.size)})</li>`;
        });
        details += '</ul>';

        details += '<p><strong>Параметры классификации:</strong></p>';
        details += `<p>Порог: ${threshold.toFixed(2)}</p>`;

        // Улучшенное отображение уровней
        const levels = [];
        if (level1) levels.push('1');
        if (level2) levels.push('2');
        if (level3) levels.push('3');
        details += `<p>Уровни: ${levels.join(', ')}</p>`;

        // Улучшенное отображение доп. опций
        const options = [];
        if (decoding) options.push('Расшифровка кодов');
        details += `<p>Доп. опции: ${options.join(' | ')}</p>`;

        document.getElementById('confirmationDetails').innerHTML = details;
        document.getElementById('confirmationModal').style.display = 'flex';

        document.getElementById('downloadBtn').style.display = 'none';

    });

    // Обработка кнопок модального окна
    document.getElementById('cancelBtn').addEventListener('click', function() {
        document.getElementById('confirmationModal').style.display = 'none';
    });

    function displayResults(results, decoding) {
        console.log('Отображение результатов:', results);
        
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('classificationResults');
        const downloadBtn = document.getElementById('downloadBtn');

        resultsSection.style.display = 'block';
        resultsContainer.innerHTML = '';
        downloadBtn.style.display = 'block'; // Показываем кнопку

        window.classificationResults = {results, decoding };
        showDownloadButton();

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            const resultHeader = document.createElement('div');
            resultHeader.className = 'result-header';
            
            const fileName = document.createElement('div');
            fileName.className = 'result-filename';
            fileName.textContent = result.filename || result.file?.name || 'Без названия';
            
            resultHeader.appendChild(fileName);
            
            const resultContent = document.createElement('div');
            resultContent.className = 'result-content';
            
            if (result.rubrics && result.rubrics.length > 0) {
                result.rubrics.forEach(rubric => {
                    const rubricItem = document.createElement('div');
                    rubricItem.className = 'rubric-item';
                    
                    const rubricCode = document.createElement('span');
                    rubricCode.className = 'rubric-code';
                    rubricCode.textContent = rubric.code || '';
                    
                    // Переносим вероятность из rubric-info на верхний уровень
                    const rubricProbability = document.createElement('span');
                    rubricProbability.className = 'rubric-probability';
                    if (rubric.probability) {
                        rubricProbability.textContent = rubric.probability.toFixed(3);
                    }
                    
                    const rubricInfo = document.createElement('div');
                    rubricInfo.className = 'rubric-info';
                    
                    if (decoding && rubric.name) {
                        const rubricName = document.createElement('span');
                        rubricName.className = 'rubric-name';
                        rubricName.textContent = rubric.name;
                        rubricInfo.appendChild(rubricName);
                    }
                    
                    rubricItem.appendChild(rubricCode);
                    rubricItem.appendChild(rubricInfo);
                    rubricItem.appendChild(rubricProbability); // Добавляем вероятность в rubric-item
                    resultContent.appendChild(rubricItem);
                });
            }
            else {
                const noResults = document.createElement('div');
                noResults.className = 'no-results';
                noResults.textContent = 'Нет рубрик, соответствующих заданному порогу';
                resultContent.appendChild(noResults);
            }
            
            resultItem.appendChild(resultHeader);
            resultItem.appendChild(resultContent);
            resultsContainer.appendChild(resultItem);
        });
    }

    document.getElementById('downloadBtn').addEventListener('click', function() {
        if (!window.classificationResults) return;
        
        const { results, decoding } = window.classificationResults;
        let csvContent = "data:text/csv;charset=utf-8,";

        let includeDecoding = document.getElementById('decoding').checked;
        csvContent += includeDecoding 
            ? "Файл;Код ГРНТИ;Название рубрики;Вероятность\n" 
            : "Файл;Код ГРНТИ;Вероятность\n";
        
    
        
        results.forEach(result => {
            const filename = result.filename || result.file?.name || 'Без названия';
            
            if (result.rubrics && result.rubrics.length > 0) {
                result.rubrics.forEach(rubric => {
                    const code = rubric.code || '';
                    const name = decoding && rubric.name ? rubric.name : '';
                    const probability = rubric.probability ? rubric.probability.toFixed(3) : '';
                    

                    csvContent += includeDecoding
                        ? `${filename};${code};"${name}";${probability}\n` 
                        : `${filename};${code};${probability}\n`;
                });
            } else {
                csvContent += includeDecoding  
                ? `${filename};"Нет рубрик, соответствующих заданному порогу";;\n` 
                : `${filename};"Нет рубрик, соответствующих заданному порогу";\n`;
            }
        });
        
        // Создаем ссылку для скачивания
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "результаты_классификации.csv");
        document.body.appendChild(link);
        
        // Запускаем скачивание
        link.click();
        
        // Удаляем ссылку
        document.body.removeChild(link);
    });
});

