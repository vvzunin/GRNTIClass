document.addEventListener('DOMContentLoaded', function() {
    // Обработка изменения положения ползунка
    const slider = document.getElementById('probabilitySlider');
    const sliderValue = document.getElementById('sliderValue');
    
    slider.addEventListener('input', function() {
        const value = this.value / 100;
        sliderValue.textContent = value.toFixed(2);
    });

    const classifyBtn = document.getElementById('classifyBtn');

    // Функции для работы с уведомлениями
    function showNotification(message) {
        const notification = document.getElementById('successNotification');
        const messageElement = document.getElementById('notificationMessage');
        
        messageElement.textContent = message;
        notification.classList.add('show');
        
        setTimeout(hideNotification, 5000);
    }
    
    function hideNotification() {
        document.getElementById('successNotification').classList.remove('show');
    }
    
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
        // const fileProgress = document.getElementById('fileProgress');
        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = message;
        
        // if (currentFile !== undefined && totalFiles) {
        //     fileProgress.textContent = `Обработано файлов: ${currentFile} из ${totalFiles}`;
        // }
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
                normalization: document.getElementById('normalization').checked,
                decoding: document.getElementById('decoding').checked,
                threshold: slider.value / 100
            };
            
            const files = fileHandler.getFiles();
            
            // Сбрасываем прогресс перед началом
            updateProgress(0, "Подготовка к обработке...", 0, files.length);
            
            const classificationResults = await API.classify(
                files, 
                params, 
                (progress, message, current, total) => {
                    updateProgress(progress, message, current, total);
                }
            );
            
            displayResults(classificationResults, params.decoding);
            loadingModal.style.display = 'none';
            
        } catch (error) {
            console.error('Ошибка классификации:', error);
            showErrorNotification(error.message);
            loadingModal.style.display = 'none';
        }
    });

    classifyBtn.addEventListener('click', function() {
        if (fileHandler.getFiles().length === 0) {
            showErrorNotification('Пожалуйста, выберите файлы для классификации');
            return;
        }
        
        const level1 = document.getElementById('level1').checked;
        const level2 = document.getElementById('level2').checked;
        const level3 = document.getElementById('level3').checked;
        const normalization = document.getElementById('normalization').checked;
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
        if (normalization) options.push('Нормализация');
        if (decoding) options.push('Расшифровка кодов');
        details += `<p>Доп. опции: ${options.join(' | ')}</p>`;

        document.getElementById('confirmationDetails').innerHTML = details;
        document.getElementById('confirmationModal').style.display = 'flex';
    });

    // Обработка кнопок модального окна
    document.getElementById('cancelBtn').addEventListener('click', function() {
        document.getElementById('confirmationModal').style.display = 'none';
    });

    function displayResults(results, decoding) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('classificationResults');
        
        resultsSection.style.display = 'block';
        resultsContainer.innerHTML = '';
        
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
            result.rubric;
            
            if (result.rubrics && result.rubrics.length > 0) {
                result.rubrics.forEach(rubric => {
                    const rubricItem = document.createElement('div');
                    rubricItem.className = 'rubric-item';
                    
                    const rubricCode = document.createElement('span');
                    rubricCode.className = 'rubric-code';
                    rubricCode.textContent = rubric.code || '';
                
                    
                    const rubricProbability = document.createElement('span');
                    rubricProbability.className = 'rubric-probability';

                    rubricProbability.textContent = '';
                    if (rubric.probability){
                        rubricProbability.textContent = (rubric.probability * 100).toFixed(1) + '%';
                    }
                    
                    
                    rubricItem.appendChild(rubricCode);
                    
                    if (decoding && rubric.name) {
                        const rubricName = document.createElement('span');
                        rubricName.className = 'rubric-name';
                        rubricName.textContent = rubric.name;
                        rubricItem.appendChild(rubricName);
                    }
                    
                    rubricItem.appendChild(rubricProbability);
                    resultContent.appendChild(rubricItem);
                });
            } else {
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
});

