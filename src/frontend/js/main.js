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

// Функция для показа уведомлений
function showNotification(message, type = 'success') {
    console.log('showNotification вызвана:', { message, type });
    
    const notification = document.getElementById(type === 'error' ? 'errorNotification' : 'successNotification');
    const messageElement = document.getElementById(type === 'error' ? 'errorMessage' : 'notificationMessage');
    
    console.log('Найденные элементы:', { notification, messageElement });
    
    if (!notification || !messageElement) {
        console.error('Не найдены элементы уведомления:', { notification, messageElement });
        return;
    }
    
    messageElement.textContent = message;
    notification.classList.add('show');
    
    console.log('Класс show добавлен, текущие классы:', notification.className);
    
    // Очищаем предыдущий таймер, если он есть
    if (notification.hideTimer) {
        clearTimeout(notification.hideTimer);
        console.log('Предыдущий таймер очищен');
    }
    
    // Устанавливаем новый таймер
    const timerId = setTimeout(() => {
        console.log('Таймер сработал, скрываем уведомление');
        console.log('Состояние уведомления до скрытия:', {
            element: notification,
            classes: notification.className,
            timer: notification.hideTimer
        });
        notification.classList.remove('show');
        notification.hideTimer = null;
        console.log('Уведомление скрыто, текущие классы:', notification.className);
        
        // Дополнительная проверка через 1 секунду
        setTimeout(() => {
            console.log('Проверка через 1 секунду - классы уведомления:', notification.className);
        }, 1000);
    }, 5000);
    
    notification.hideTimer = timerId;
    console.log(`Показано уведомление: ${type} - ${message}, таймер установлен на 5 секунд, ID: ${timerId}`);
}

// Глобальные функции для уведомлений
window.showErrorNotification = (message) => showNotification(message, 'error');
window.showSuccessNotification = (message) => showNotification(message, 'success');

// Функция для скрытия всех уведомлений
function hideAllNotifications() {
    const notifications = document.querySelectorAll('.notification');
    notifications.forEach(notification => {
        notification.classList.remove('show');
        if (notification.hideTimer) {
            clearTimeout(notification.hideTimer);
            notification.hideTimer = null;
        }
    });
}

window.hideAllNotifications = hideAllNotifications;

// Функция для проверки стилей уведомлений
function checkNotificationStyles() {
    const successNotification = document.getElementById('successNotification');
    const errorNotification = document.getElementById('errorNotification');
    
    if (successNotification) {
        const styles = window.getComputedStyle(successNotification);
        console.log('Стили successNotification:', {
            display: styles.display,
            transform: styles.transform,
            transition: styles.transition,
            zIndex: styles.zIndex
        });
    }
    
    if (errorNotification) {
        const styles = window.getComputedStyle(errorNotification);
        console.log('Стили errorNotification:', {
            display: styles.display,
            transform: styles.transform,
            transition: styles.transition,
            zIndex: styles.zIndex
        });
    }
}

window.checkNotificationStyles = checkNotificationStyles;

// Функция для принудительного скрытия уведомлений (для тестирования)
function forceHideNotifications() {
    console.log('Принудительное скрытие всех уведомлений...');
    const notifications = document.querySelectorAll('.notification');
    notifications.forEach((notification, index) => {
        console.log(`Скрываем уведомление ${index + 1}:`, notification);
        notification.classList.remove('show');
        if (notification.hideTimer) {
            clearTimeout(notification.hideTimer);
            notification.hideTimer = null;
            console.log(`Таймер ${notification.hideTimer} очищен для уведомления ${index + 1}`);
        }
    });
    console.log('Все уведомления скрыты');
}

window.forceHideNotifications = forceHideNotifications;

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM загружен, инициализируем приложение...');
    
    // Проверяем стили уведомлений
    checkNotificationStyles();
    
    // Делаем функцию проверки состояния сервера доступной глобально
    window.initializeServerHealthCheck = initializeServerHealthCheck;
    
    // Проверяем состояние сервера только если конфигурация уже загружена
    if (window.configLoaded) {
        initializeServerHealthCheck();
    }

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

    function updateProgress(progress, message, currentFile, totalFiles) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const fileProgress = document.getElementById('fileProgress');
        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = message;
        
        if (totalFiles > 1) {
            // Показываем информацию о количестве файлов и этапе обработки
            // Файлы обрабатываются батчами на сервере, поэтому показываем общий прогресс
            if (progress < 20) {
                fileProgress.textContent = `Подготовка к классификации ${totalFiles} файлов...`;
            } else if (progress >= 90) {
                fileProgress.textContent = `Завершение классификации...`;
            } else {
                fileProgress.textContent = `Классификация ${totalFiles} файлов...`;
            }
        } else {
            fileProgress.textContent = '';
        }
    }

    // Функция инициализации проверки состояния сервера
    async function initializeServerHealthCheck() {
        // Если проверка отключена, не выполняем её
        if (!API.config.enableHealthCheck) {
            console.log('Проверка состояния сервера отключена');
            return;
        }
        
        // Проверяем, что конфигурация загружена
        if (!window.configLoaded) {
            console.log('Конфигурация еще не загружена, пропускаем проверку состояния сервера');
            return;
        }
        
        try {
            const health = await API.checkServerHealth();
            if (health.status === 'error') {
                // Показываем предупреждение только в консоли, не беспокоим пользователя
                console.warn('Предупреждение о состоянии сервера:', health.message);
                // Показываем индикатор только для реальных ошибок
                showServerStatusIndicator('error', health.message);
            } else {
                console.log('Сервер работает нормально');
                // Не показываем индикатор для успешного состояния
            }
        } catch (error) {
            console.warn('Не удалось проверить состояние сервера:', error);
            showServerStatusIndicator('unknown', 'Статус сервера неизвестен');
        }
    }

    // Функция для отображения индикатора состояния сервера
    function showServerStatusIndicator(status, message) {
        // Показываем индикатор только для ошибок и предупреждений
        if (status === 'healthy') {
            console.log('Сервер работает нормально');
            return; // Не показываем индикатор для успешного состояния
        }
        
        // Создаем или обновляем индикатор состояния сервера
        let indicator = document.getElementById('serverStatusIndicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'serverStatusIndicator';
            indicator.style.cssText = `
                position: fixed;
                top: 10px;
                left: 10px;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                z-index: 1000;
                opacity: 0.8;
                transition: opacity 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            `;
            document.body.appendChild(indicator);
        }
        
        const colors = {
            healthy: '#28a745',
            warning: '#ffc107',
            error: '#dc3545',
            unknown: '#6c757d'
        };
        
        indicator.style.backgroundColor = colors[status] || colors.unknown;
        indicator.style.color = status === 'warning' ? '#000' : '#fff';
        indicator.textContent = `Сервер: ${message}`;
        
        // Автоматически скрываем через 3 секунды для ошибок
        const hideDelay = status === 'error' ? 3000 : 5000;
        setTimeout(() => {
            indicator.style.opacity = '0';
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.parentNode.removeChild(indicator);
                }
            }, 300);
        }, hideDelay);
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
            
            // Проверяем состояние сервера перед отправкой (мягкая проверка)
            try {
                const health = await API.checkServerHealth();
                if (health.status === 'error') {
                    console.warn('Предупреждение: сервер может быть недоступен:', health.message);
                    // Не прерываем процесс, просто предупреждаем
                }
            } catch (error) {
                console.warn('Не удалось проверить состояние сервера перед отправкой:', error);
                // Продолжаем процесс, так как основная классификация может работать
            }
            
            // Скрываем предыдущие уведомления
            hideAllNotifications();
            
            // Сбрасываем прогресс перед началом
            updateProgress(0, "Подготовка к классификации...", 0, files.length);
            
            console.log('Начинаем классификацию...');
            const classificationResults = await API.classify(
                files, 
                params, 
                (progress, message, current, total) => {
                    updateProgress(progress, message, current, total);
                }
            );
            
            console.log('Результаты получены:', classificationResults);
            
            // Сначала скрываем модальное окно загрузки
            loadingModal.style.display = 'none';
            
            // Уведомление о завершении классификации убрано
            
            // И только потом отображаем результаты (с небольшой задержкой)
            setTimeout(() => {
                displayResults(classificationResults, params.decoding);
            }, 100);
            
        } catch (error) {
            console.error('Ошибка классификации:', error);
            showErrorNotification(error.message);
            loadingModal.style.display = 'none';
        }
    });

    classifyBtn.addEventListener('click', function() {
        const files = fileHandler.getFiles();
        
        if (files.length === 0) {
            showErrorNotification('Пожалуйста, выберите файлы для обработки');
            return;
        }
        
        const level1 = document.getElementById('level1').checked;
        const level2 = document.getElementById('level2').checked;
        const level3 = document.getElementById('level3').checked;
        const decoding = document.getElementById('decoding').checked;
        const threshold = slider.value / 100;

        // Проверяем, что выбран хотя бы один уровень
        if (!level1 && !level2 && !level3) {
            showErrorNotification('Пожалуйста, выберите хотя бы один уровень классификации');
            return;
        }

        // Формируем текст подтверждения
        let details = '<p><strong>Выбранные файлы:</strong></p><ul>';
        
        // Показываем только количество файлов
        details += `<li>Количество файлов: ${files.length}</li>`;
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
        details += `<p>Доп. опции: ${options.join(' | ') || 'Нет'}</p>`;

        document.getElementById('confirmationDetails').innerHTML = details;
        document.getElementById('confirmationModal').style.display = 'flex';

        document.getElementById('downloadBtn').style.display = 'none';
    });

    // Обработка кнопок модального окна
    document.getElementById('cancelBtn').addEventListener('click', function() {
        document.getElementById('confirmationModal').style.display = 'none';
    });

    // Добавляем кнопку очистки всех файлов
    const clearFilesBtn = document.createElement('button');
    clearFilesBtn.textContent = 'Удалить все файлы';
    clearFilesBtn.className = 'clear-files-btn';
    clearFilesBtn.addEventListener('click', function() {
        if (fileHandler.getFiles().length > 0) {
            fileHandler.clearFiles();
            showSuccessNotification('Все файлы удалены');
        }
    });
    
    // Добавляем кнопку в DOM
    const fileUploadArea = document.getElementById('fileUploadArea');
    fileUploadArea.appendChild(clearFilesBtn);

    function displayResults(results, decoding) {
        console.log('Отображение результатов:', results);
        
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('classificationResults');
        const downloadBtn = document.getElementById('downloadBtn');

        resultsSection.style.display = 'block';
        resultsContainer.innerHTML = '';
        downloadBtn.style.display = 'block';

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
            
            // Добавляем индикатор ошибки
            if (result.error) {
                const errorIndicator = document.createElement('span');
                errorIndicator.className = 'error-indicator';
                errorIndicator.textContent = 'Ошибка';
                errorIndicator.title = result.error;
                resultHeader.appendChild(errorIndicator);
            }
            
            resultHeader.appendChild(fileName);
            
            const resultContent = document.createElement('div');
            resultContent.className = 'result-content';
            
            if (result.error) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = `Ошибка: ${result.error}`;
                resultContent.appendChild(errorDiv);
            } else if (result.rubrics && result.rubrics.length > 0) {
                result.rubrics.forEach(rubric => {
                    const rubricItem = document.createElement('div');
                    rubricItem.className = 'rubric-item';
                    
                    const rubricCode = document.createElement('span');
                    rubricCode.className = 'rubric-code';
                    rubricCode.textContent = rubric.code || '';
                    
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
            
            if (result.error) {
                csvContent += includeDecoding  
                    ? `${filename};"Ошибка обработки";"${result.error}";\n` 
                    : `${filename};"Ошибка обработки";\n`;
            } else if (result.rubrics && result.rubrics.length > 0) {
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
        
        showSuccessNotification('Файл с результатами скачан');
    });

    // Добавляем обработчик для закрытия уведомлений
    const closeButtons = document.querySelectorAll('.notification-close');
    console.log('Найдено кнопок закрытия:', closeButtons.length);
    
    closeButtons.forEach((closeBtn, index) => {
        console.log(`Добавляем обработчик для кнопки ${index + 1}`);
        closeBtn.addEventListener('click', function() {
            console.log('Кнопка закрытия нажата');
            const notification = this.parentElement;
            notification.classList.remove('show');
            
            // Очищаем таймер, если он есть
            if (notification.hideTimer) {
                clearTimeout(notification.hideTimer);
                notification.hideTimer = null;
                console.log('Таймер очищен при ручном закрытии');
            }
            
            console.log('Уведомление закрыто пользователем');
        });
    });
});

