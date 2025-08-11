class API {
    static config = {
        maxRetries: 3,
        retryDelay: 1000,
        timeout: 300000, // 5 минут
        chunkSize: 1024 * 1024, // 1MB для больших файлов
        enableCompression: true, // Возможность отключения сжатия
        compressionThreshold: 5 * 1024 * 1024, // Порог для сжатия (5MB)
        enableHealthCheck: true, // Возможность отключения проверки состояния сервера
        healthCheckTimeout: 5000 // Таймаут для проверки состояния сервера
    };

    // Асинхронная функция для работы с конфигурацией
    static async getConfig() {
        try {
            const response = await fetch('config.json');
            return await response.json();
        } catch (error) {
            console.error('Ошибка загрузки конфигурации:', error);
            return null;
        }
    }

    // Сжатие файлов перед отправкой
    static async compressFiles(files) {
        const compressedFiles = [];
        
        for (const file of files) {
            try {
                // Для текстовых файлов можно применить простое сжатие
                const content = await this.readFileContent(file);
                const compressedContent = this.compressText(content);
                
                // Создаем новый файл с сжатым содержимым
                const compressedFile = new File(
                    [compressedContent], 
                    file.name, 
                    { type: file.type }
                );
                
                compressedFiles.push(compressedFile);
            } catch (error) {
                console.warn('Не удалось сжать файл:', file.name, error);
                compressedFiles.push(file); // Используем оригинальный файл
            }
        }
        
        return compressedFiles;
    }

    // Простое сжатие текста (удаление лишних пробелов, переносов строк)
    static compressText(text) {
        return text
            .replace(/\s+/g, ' ') // Заменяем множественные пробелы на один
            .replace(/\n\s*\n/g, '\n') // Убираем пустые строки
            .trim();
    }

    // Чтение содержимого файла
    static readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = e => reject(new Error('Ошибка чтения файла'));
            reader.readAsText(file, 'UTF-8');
        });
    }

    // Отправка файлов с реальным прогрессом
    static async classify(files, params, progressCallback) {
        let retryCount = 0;
        
        while (retryCount < this.config.maxRetries) {
            try {
                return await this.performClassification(files, params, progressCallback);
            } catch (error) {
                retryCount++;
                console.warn(`Попытка ${retryCount} не удалась:`, error);
                
                if (retryCount >= this.config.maxRetries) {
                    throw new Error(`Не удалось выполнить классификацию после ${this.config.maxRetries} попыток: ${error.message}`);
                }
                
                // Ждем перед повторной попыткой (только для реальных сетевых ошибок)
                if (error.name === 'TypeError' || error.name === 'AbortError') {
                    await this.delay(this.config.retryDelay * retryCount);
                }
                progressCallback(0, `Повторная попытка ${retryCount}...`, 0, files.length);
            }
        }
    }

    // Основная функция классификации
    static async performClassification(files, params, progressCallback) {
        return new Promise(async (resolve, reject) => {
            try {
                // Подготовка файлов
                progressCallback(5, "Подготовка к классификации...", 0, files.length);
                
                // Сжимаем файлы если они большие и сжатие включено
                const totalSize = files.reduce((sum, file) => sum + file.size, 0);
                const shouldCompress = this.config.enableCompression && 
                                     totalSize > this.config.compressionThreshold;
                
                const filesToSend = shouldCompress ? 
                    await this.compressFiles(files) : files;
                
                progressCallback(10, "Подготовка данных для отправки...", 0, files.length);
                
                const formData = new FormData();
                
                filesToSend.forEach(file => formData.append('files', file));
                formData.append('level1', params.level1);
                formData.append('level2', params.level2);
                formData.append('level3', params.level3);
                formData.append('decoding', params.decoding);
                formData.append('threshold', params.threshold);

                // Создаем контроллер для отмены запроса
                const controller = new AbortController();
                const timeoutId = setTimeout(() => {
                    controller.abort();
                }, this.config.timeout);

                progressCallback(15, "Отправка файлов на сервер для классификации...", 0, files.length);
                
                console.log('Отправляем запрос на:', window.apiUrl);
                console.log('Количество файлов:', files.length);
                console.log('Общий размер:', this.formatFileSize(totalSize));
                if (shouldCompress) {
                    console.log('Применено сжатие файлов для оптимизации передачи');
                }
                
                const response = await fetch(window.apiUrl, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                console.log('Получен ответ:', response.status, response.statusText);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
                }
                
                // Показываем этапы обработки с адаптивными задержками
                // Примечание: файлы обрабатываются батчами на сервере
                const startTime = Date.now();
                const batchSize = files.reduce((sum, file) => sum + file.size, 0);
                const isLargeBatch = files.length > 3 || batchSize > 10 * 1024 * 1024; // >3 файлов или >10MB
                
                const stages = [
                    { progress: 20, message: "Отправка файлов на сервер..." },
                    { progress: 40, message: "Анализ содержимого файлов..." },
                    { progress: 60, message: "Классификация по рубрикам ГРНТИ..." },
                    { progress: 80, message: "Подготовка результатов..." }
                ];
                
                for (let i = 0; i < stages.length; i++) {
                    const stage = stages[i];
                    progressCallback(stage.progress, stage.message, 0, files.length);
                    
                    // Адаптивная задержка: учитываем размер батча и время обработки
                    if (i < stages.length - 1) {
                        const elapsed = Date.now() - startTime;
                        
                        if (isLargeBatch) {
                            // Для больших батчей - минимальные задержки
                            const minDelay = 100; // Быстрое переключение этапов
                            const adaptiveDelay = Math.max(0, minDelay - elapsed / stages.length);
                            
                            if (adaptiveDelay > 0) {
                                await this.delay(adaptiveDelay);
                            }
                        } else {
                            // Для маленьких батчей - показываем этапы достаточно долго
                            const minDelay = 400; // Больше времени на чтение
                            const maxDelay = 800; // Но не слишком долго
                            
                            const adaptiveDelay = Math.max(0, minDelay - elapsed / stages.length);
                            const finalDelay = Math.min(adaptiveDelay, maxDelay);
                            
                            if (finalDelay > 0) {
                                await this.delay(finalDelay);
                            }
                        }
                    }
                }
                
                progressCallback(90, "Получение результатов классификации...", files.length, files.length);
                
                const data = await response.json();
                
                progressCallback(100, "Классификация завершена", files.length, files.length);
                
                if (data.type === 'error') {
                    throw new Error(data.message);
                }
                
                if (data.type === 'result') {
                    const results = data.results || [];
                    console.log('Получены результаты:', results);
                    resolve(results);
                } else {
                    console.error('Неожиданный формат ответа:', data);
                    throw new Error('Неожиданный формат ответа от сервера');
                }
                
            } catch (error) {
                console.error('Ошибка API:', error);
                
                if (error.name === 'AbortError') {
                    reject(new Error('Превышено время ожидания ответа от сервера'));
                } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    reject(new Error('Ошибка соединения с сервером. Проверьте подключение к интернету'));
                } else {
                    reject(new Error(`Ошибка обработки: ${error.message}`));
                }
            }
        });
    }

    // Отправка файлов по частям для больших файлов
    static async uploadLargeFiles(files, params, progressCallback) {
        const results = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            progressCallback(
                (i / files.length) * 100, 
                `Обработка файла ${i + 1} из ${files.length}: ${file.name}`, 
                i, 
                files.length
            );
            
            try {
                const result = await this.uploadSingleFile(file, params);
                results.push(result);
            } catch (error) {
                console.error(`Ошибка обработки файла ${file.name}:`, error);
                results.push({
                    filename: file.name,
                    error: error.message,
                    rubrics: []
                });
            }
        }
        
        return results;
    }

    // Отправка одного файла
    static async uploadSingleFile(file, params) {
        const formData = new FormData();
        formData.append('files', file);
        formData.append('level1', params.level1);
        formData.append('level2', params.level2);
        formData.append('level3', params.level3);
        formData.append('decoding', params.decoding);
        formData.append('threshold', params.threshold);

        const response = await fetch(window.apiUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.type === 'error') {
            throw new Error(data.message);
        }
        
        return data.results?.[0] || { filename: file.name, rubrics: [] };
    }

    // Проверка состояния сервера
    static async checkServerHealth() {
        // Если проверка отключена, возвращаем успех
        if (!this.config.enableHealthCheck) {
            return {
                status: 'healthy',
                message: 'Проверка состояния сервера отключена'
            };
        }
        
        // Проверяем, что URL сервера доступен
        if (!window.apiUrl) {
            return {
                status: 'error',
                message: 'URL сервера не настроен'
            };
        }
        
        try {
            // Сначала пробуем эндпоинт /health
            const healthUrl = window.apiUrl.replace('/classify', '/health');
            const response = await fetch(healthUrl, {
                method: 'GET',
                signal: AbortSignal.timeout(this.config.healthCheckTimeout)
            });
            
            if (response.ok) {
                const data = await response.json();
                return {
                    status: 'healthy',
                    message: data.message || 'Сервер работает нормально'
                };
            } else {
                // Если /health недоступен, пробуем корневой эндпоинт
                const rootUrl = window.apiUrl.replace('/classify', '/');
                const rootResponse = await fetch(rootUrl, {
                    method: 'GET',
                    signal: AbortSignal.timeout(this.config.healthCheckTimeout)
                });
                
                if (rootResponse.ok) {
                    return {
                        status: 'healthy',
                        message: 'Сервер работает нормально'
                    };
                } else {
                    return {
                        status: 'error',
                        message: `Сервер недоступен: ${rootResponse.status}`
                    };
                }
            }
            
        } catch (error) {
            // Если ошибка сети или таймаут
            if (error.name === 'AbortError') {
                return {
                    status: 'error',
                    message: 'Сервер не отвечает (таймаут)'
                };
            }
            
            // Если ошибка связана с методом (405), сервер работает
            if (error.name === 'TypeError' && error.message.includes('405')) {
                return {
                    status: 'healthy',
                    message: 'Сервер работает нормально'
                };
            }
            
            return {
                status: 'error',
                message: 'Не удается подключиться к серверу'
            };
        }
    }

    // Утилиты
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    static formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Получение статистики файлов
    static getFilesStats(files) {
        const totalSize = files.reduce((sum, file) => sum + file.size, 0);
        const avgSize = files.length > 0 ? totalSize / files.length : 0;
        
        return {
            count: files.length,
            totalSize,
            avgSize,
            largestFile: files.reduce((max, file) => file.size > max.size ? file : max, { size: 0 }),
            smallestFile: files.reduce((min, file) => file.size < min.size ? file : min, { size: Infinity })
        };
    }
}

window.API = API;