class API {
// Асинхронная функция для работы с конфигурацией


    static async classify(files, params, progressCallback) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            
            files.forEach(file => formData.append('files', file));
            formData.append('level1', params.level1);
            formData.append('level2', params.level2);
            formData.append('level3', params.level3);
            // formData.append('normalization', params.normalization);
            formData.append('decoding', params.decoding);
            formData.append('threshold', params.threshold);

            // Имитация прогресса для пользователя
            progressCallback(0, "Отправка файлов на сервер...", 0, files.length);
            
            setTimeout(() => {
                progressCallback(25, "Обработка файлов...", 0, files.length);
            }, 500);
            
            setTimeout(() => {
                progressCallback(50, "Классификация по уровням ГРНТИ...", 0, files.length);
            }, 1000);
            
            setTimeout(() => {
                progressCallback(75, "Формирование результатов...", 0, files.length);
            }, 2000);

            console.log('Отправляем запрос на:', window.apiUrl);
            console.log('Данные формы:', formData);
            
            fetch(window.apiUrl, {
                method: 'POST',
                body: formData
            }).then(response => {
                console.log('Получен ответ:', response.status, response.statusText);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return response.json();
            }).then(data => {
                progressCallback(100, "Обработка завершена", files.length, files.length);
                
                if (data.type === 'error') {
                    throw new Error(data.message);
                }
                
                if (data.type === 'result') {
                    // Преобразуем результат в формат, ожидаемый фронтендом
                    const results = data.results || [];
                    console.log('Получены результаты:', results);
                    resolve(results);
                } else {
                    console.error('Неожиданный формат ответа:', data);
                    throw new Error('Неожиданный формат ответа от сервера');
                }
            }).catch(error => {
                console.error('Ошибка API:', error);
                reject(new Error(`Ошибка соединения: ${error.message}`));
            });
        });
    }
}
window.API = API;