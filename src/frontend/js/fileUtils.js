/**
 * Утилиты для работы с файлами
 */
class FileUtils {
    // Константы для валидации
    static MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    static MAX_FILES_COUNT = 50;
    static SUPPORTED_TYPES = ['text/plain'];
    static SUPPORTED_EXTENSIONS = ['.txt'];
    static MAX_CONTENT_LENGTH = 1000000; // 1MB текста

    /**
     * Валидация файла
     * @param {File} file - Файл для валидации
     * @returns {Object} Результат валидации
     */
    static validateFile(file) {
        const errors = [];
        const warnings = [];

        // Проверка размера
        if (file.size > this.MAX_FILE_SIZE) {
            errors.push(`Файл "${file.name}" слишком большой (${this.formatFileSize(file.size)}). Максимальный размер: ${this.formatFileSize(this.MAX_FILE_SIZE)}`);
        }

        // Проверка типа файла
        const isValidType = this.SUPPORTED_TYPES.includes(file.type) || 
                           this.SUPPORTED_EXTENSIONS.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValidType) {
            errors.push(`Файл "${file.name}" имеет неподдерживаемый формат. Поддерживаются только .txt файлы`);
        }

        // Проверка на пустой файл
        if (file.size === 0) {
            errors.push(`Файл "${file.name}" пустой`);
        }

        // Предупреждения
        if (file.size > 1 * 1024 * 1024) { // 1MB
            warnings.push(`Файл "${file.name}" довольно большой и может обрабатываться медленно`);
        }

        return {
            isValid: errors.length === 0,
            errors,
            warnings
        };
    }

    /**
     * Валидация списка файлов
     * @param {File[]} files - Список файлов
     * @returns {Object} Результат валидации
     */
    static validateFiles(files) {
        const errors = [];
        const warnings = [];
        const validFiles = [];

        // Проверка количества файлов
        if (files.length > this.MAX_FILES_COUNT) {
            errors.push(`Слишком много файлов: ${files.length}. Максимальное количество: ${this.MAX_FILES_COUNT}`);
        }

        // Проверка каждого файла
        files.forEach(file => {
            const validation = this.validateFile(file);
            errors.push(...validation.errors);
            warnings.push(...validation.warnings);
            
            if (validation.isValid) {
                validFiles.push(file);
            }
        });

        return {
            isValid: errors.length === 0,
            errors,
            warnings,
            validFiles
        };
    }

    /**
     * Анализ содержимого файла
     * @param {File} file - Файл для анализа
     * @returns {Promise<Object>} Результат анализа
     */
    static async analyzeFile(file) {
        try {
            const content = await this.readFileContent(file);
            
            const lines = content.split('\n');
            const nonEmptyLines = lines.filter(line => line.trim());
            
            const words = content.split(/\s+/).filter(word => word.trim());
            const uniqueWords = new Set(words.map(word => word.toLowerCase()));
            
            // Подсчет символов разных типов
            const charCount = content.length;
            const letterCount = (content.match(/[а-яёa-z]/gi) || []).length;
            const digitCount = (content.match(/\d/g) || []).length;
            const spaceCount = (content.match(/\s/g) || []).length;
            
            // Определение языка (простая эвристика)
            const cyrillicChars = (content.match(/[а-яё]/gi) || []).length;
            const latinChars = (content.match(/[a-z]/gi) || []).length;
            const language = cyrillicChars > latinChars ? 'russian' : 'english';
            
            return {
                lineCount: lines.length,
                nonEmptyLineCount: nonEmptyLines.length,
                wordCount: words.length,
                uniqueWordCount: uniqueWords.size,
                charCount,
                letterCount,
                digitCount,
                spaceCount,
                language,
                avgLineLength: lines.length > 0 ? charCount / lines.length : 0,
                avgWordLength: words.length > 0 ? letterCount / words.length : 0,
                preview: nonEmptyLines.slice(0, 3).join('\n') + (nonEmptyLines.length > 3 ? '\n...' : ''),
                encoding: this.detectEncoding(content)
            };
        } catch (error) {
            console.error('Ошибка анализа файла:', error);
            return null;
        }
    }

    /**
     * Чтение содержимого файла с обработкой ошибок
     * @param {File} file - Файл для чтения
     * @returns {Promise<string>} Содержимое файла
     */
    static readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = e => {
                try {
                    const content = e.target.result;
                    
                    if (content.length === 0) {
                        reject(new Error('Файл пустой'));
                        return;
                    }
                    
                    if (content.length > this.MAX_CONTENT_LENGTH) {
                        reject(new Error('Файл слишком большой для обработки'));
                        return;
                    }
                    
                    // Проверка на наличие нечитаемых символов
                    const hasInvalidChars = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content);
                    if (hasInvalidChars) {
                        console.warn('Файл содержит нечитаемые символы:', file.name);
                    }
                    
                    resolve(content);
                } catch (error) {
                    reject(new Error('Ошибка обработки содержимого файла'));
                }
            };
            
            reader.onerror = e => reject(new Error('Ошибка чтения файла'));
            reader.onabort = e => reject(new Error('Чтение файла прервано'));
            
            // Устанавливаем таймаут
            const timeout = setTimeout(() => {
                reader.abort();
                reject(new Error('Превышено время чтения файла'));
            }, 30000);
            
            reader.onloadend = () => clearTimeout(timeout);
            
            reader.readAsText(file, 'UTF-8');
        });
    }

    /**
     * Определение кодировки файла
     * @param {string} content - Содержимое файла
     * @returns {string} Кодировка
     */
    static detectEncoding(content) {
        // Простая эвристика для определения кодировки
        const hasCyrillic = /[а-яё]/i.test(content);
        const hasLatin = /[a-z]/i.test(content);
        
        if (hasCyrillic && !hasLatin) {
            return 'UTF-8 (Cyrillic)';
        } else if (hasLatin && !hasCyrillic) {
            return 'UTF-8 (Latin)';
        } else if (hasCyrillic && hasLatin) {
            return 'UTF-8 (Mixed)';
        } else {
            return 'UTF-8 (Unknown)';
        }
    }

    /**
     * Сжатие текста
     * @param {string} text - Исходный текст
     * @returns {string} Сжатый текст
     */
    static compressText(text) {
        return text
            .replace(/\s+/g, ' ') // Заменяем множественные пробелы на один
            .replace(/\n\s*\n/g, '\n') // Убираем пустые строки
            .replace(/\t/g, ' ') // Заменяем табуляцию на пробелы
            .trim();
    }

    /**
     * Создание сжатого файла
     * @param {File} file - Исходный файл
     * @returns {Promise<File>} Сжатый файл
     */
    static async createCompressedFile(file) {
        try {
            const content = await this.readFileContent(file);
            const compressedContent = this.compressText(content);
            
            return new File(
                [compressedContent], 
                file.name, 
                { type: file.type }
            );
        } catch (error) {
            console.warn('Не удалось сжать файл:', file.name, error);
            return file; // Возвращаем оригинальный файл
        }
    }

    /**
     * Форматирование размера файла
     * @param {number} bytes - Размер в байтах
     * @returns {string} Отформатированный размер
     */
    static formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    }

    /**
     * Получение статистики файлов
     * @param {File[]} files - Список файлов
     * @returns {Object} Статистика
     */
    static getFilesStats(files) {
        if (files.length === 0) {
            return {
                count: 0,
                totalSize: 0,
                avgSize: 0,
                largestFile: null,
                smallestFile: null,
                sizeDistribution: {}
            };
        }

        const totalSize = files.reduce((sum, file) => sum + file.size, 0);
        const avgSize = totalSize / files.length;
        
        const largestFile = files.reduce((max, file) => file.size > max.size ? file : max, files[0]);
        const smallestFile = files.reduce((min, file) => file.size < min.size ? file : min, files[0]);

        // Распределение по размерам
        const sizeDistribution = {
            small: files.filter(f => f.size < 1024).length, // < 1KB
            medium: files.filter(f => f.size >= 1024 && f.size < 1024 * 1024).length, // 1KB - 1MB
            large: files.filter(f => f.size >= 1024 * 1024).length // > 1MB
        };

        return {
            count: files.length,
            totalSize,
            avgSize,
            largestFile,
            smallestFile,
            sizeDistribution
        };
    }

    /**
     * Создание уникального имени файла
     * @param {string} originalName - Оригинальное имя
     * @param {number} index - Индекс
     * @returns {string} Уникальное имя
     */
    static createUniqueFileName(originalName, index = 0) {
        const name = originalName.replace(/\.[^/.]+$/, ''); // Убираем расширение
        const extension = originalName.match(/\.[^/.]+$/)?.[0] || '';
        
        if (index === 0) {
            return originalName;
        }
        
        return `${name}_${index}${extension}`;
    }

    /**
     * Проверка дубликатов файлов
     * @param {File[]} files - Список файлов
     * @returns {Object} Информация о дубликатах
     */
    static findDuplicates(files) {
        const duplicates = [];
        const seen = new Map();

        files.forEach((file, index) => {
            const key = `${file.name}_${file.size}`;
            
            if (seen.has(key)) {
                duplicates.push({
                    original: seen.get(key),
                    duplicate: index,
                    file: file
                });
            } else {
                seen.set(key, index);
            }
        });

        return {
            hasDuplicates: duplicates.length > 0,
            duplicates,
            count: duplicates.length
        };
    }

    /**
     * Экспорт статистики в JSON
     * @param {File[]} files - Список файлов
     * @returns {string} JSON строка
     */
    static exportStatsToJSON(files) {
        const stats = this.getFilesStats(files);
        const duplicates = this.findDuplicates(files);
        
        return JSON.stringify({
            timestamp: new Date().toISOString(),
            filesCount: stats.count,
            totalSize: stats.totalSize,
            averageSize: stats.avgSize,
            sizeDistribution: stats.sizeDistribution,
            duplicates: duplicates,
            files: files.map(file => ({
                name: file.name,
                size: file.size,
                type: file.type,
                lastModified: file.lastModified
            }))
        }, null, 2);
    }
}

// Делаем доступным глобально
window.FileUtils = FileUtils;
