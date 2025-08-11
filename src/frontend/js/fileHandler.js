class FileHandler {
    constructor() {
      this.files = [];
      this.maxFileSize = 10 * 1024 * 1024; // 10MB
      this.maxFiles = 50;
      this.supportedTypes = ['text/plain'];
      this.supportedExtensions = ['.txt'];
      
      this.fileInput = document.getElementById('fileInput');
      this.fileStatus = document.getElementById('fileStatus');
      this.fileList = document.getElementById('fileList');
      this.fileUploadArea = document.getElementById('fileUploadArea');
      
      this.initEvents();
    }
  
    // Инициализация всех обработчиков событий
    initEvents() {
      this.fileInput.addEventListener('change', () => this.handleFileSelect());
      this.fileUploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
      this.fileUploadArea.addEventListener('dragleave', () => this.handleDragLeave());
      this.fileUploadArea.addEventListener('drop', (e) => this.handleDrop(e));
      
      // Добавляем обработчик для очистки при клике на область
      this.fileUploadArea.addEventListener('click', (e) => {
        if (e.target === this.fileUploadArea) {
          this.fileInput.click();
        }
      });
    }

    // Валидация файла
    validateFile(file) {
      const errors = [];
      
      // Проверка размера
      if (file.size > this.maxFileSize) {
        errors.push(`Файл "${file.name}" слишком большой (${this.formatFileSize(file.size)}). Максимальный размер: ${this.formatFileSize(this.maxFileSize)}`);
      }
      
      // Проверка типа файла
      const isValidType = this.supportedTypes.includes(file.type) || 
                         this.supportedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      
      if (!isValidType) {
        errors.push(`Файл "${file.name}" имеет неподдерживаемый формат. Поддерживаются только .txt файлы`);
      }
      
      // Проверка на пустой файл
      if (file.size === 0) {
        errors.push(`Файл "${file.name}" пустой`);
      }
      
      return {
        isValid: errors.length === 0,
        errors
      };
    }

    // Показ уведомлений об ошибках
    showValidationErrors(errors) {
      const errorMessage = errors.join('\n');
      if (window.showErrorNotification) {
        window.showErrorNotification(errorMessage);
      } else {
        alert(errorMessage);
      }
    }

    // Предварительное чтение файла для проверки содержимого
    async previewFile(file) {
      try {
        const content = await this.readFileContent(file);
        const lines = content.split('\n').filter(line => line.trim());
        
        return {
          lineCount: lines.length,
          wordCount: content.split(/\s+/).filter(word => word.trim()).length,
          charCount: content.length,
          preview: lines.slice(0, 3).join('\n') + (lines.length > 3 ? '\n...' : '')
        };
      } catch (error) {
        console.error('Ошибка предварительного чтения файла:', error);
        return null;
      }
    }

    handleFileSelect() {
        return new Promise(async (resolve) => {
          const selectedFiles = Array.from(this.fileInput.files);
          await this.processFiles(selectedFiles);
          hideDownloadButton();
          resolve();
        });
    }
  
    // Обработка перетаскивания файлов (над областью)
    handleDragOver(e) {
      e.preventDefault();
      e.stopPropagation();
      this.fileUploadArea.classList.add('drag-over');
    }
  
    // Обработка когда файлы ушли за пределы области
    handleDragLeave(e) {
      e.preventDefault();
      e.stopPropagation();
      // Проверяем, что мы действительно покинули область, а не перешли к дочернему элементу
      if (!this.fileUploadArea.contains(e.relatedTarget)) {
        this.fileUploadArea.classList.remove('drag-over');
      }
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.fileUploadArea.classList.remove('drag-over');
        
        return new Promise(async (resolve) => {
          const droppedFiles = Array.from(e.dataTransfer.files);
          await this.processFiles(droppedFiles);
          hideDownloadButton();
          resolve();
        });
    }

    // Обработка файлов с валидацией
    async processFiles(newFiles) {
      const validFiles = [];
      const allErrors = [];
      
      // Проверяем лимит на количество файлов
      if (this.files.length + newFiles.length > this.maxFiles) {
        allErrors.push(`Максимальное количество файлов: ${this.maxFiles}. Текущее: ${this.files.length}, пытаетесь добавить: ${newFiles.length}`);
      }
      
      // Валидируем каждый файл
      for (const file of newFiles) {
        const validation = this.validateFile(file);
        if (validation.isValid) {
          validFiles.push(file);
        } else {
          allErrors.push(...validation.errors);
        }
      }
      
      // Показываем ошибки, если есть
      if (allErrors.length > 0) {
        this.showValidationErrors(allErrors);
      }
      
      // Добавляем валидные файлы
      if (validFiles.length > 0) {
        this.files.push(...validFiles);
        this.updateFileDisplay();
      }
    }
  
    // Обновление отображения информации о файлах
    async updateFileDisplay() {
      document.getElementById('resultsSection').style.display = 'none';
      
      if (this.files.length === 0) {
        this.fileStatus.textContent = 'Файлы не выбраны';
        this.fileList.innerHTML = '';
        this.fileList.classList.add('empty');
        document.getElementById('classifyBtn').disabled = true;
      } else {
        this.fileStatus.textContent = this.getFilesCountText();
        await this.renderFileList();
        document.getElementById('classifyBtn').disabled = false;
      }
    }
  
    // Формирование текста с количеством файлов
    getFilesCountText() {
      const count = this.files.length;
      const totalSize = this.files.reduce((sum, file) => sum + file.size, 0);
      
      let fileWord;
      let chooseWord = '';
  
      if (count % 10 === 1 && count % 100 !== 11) {
        fileWord = 'файл';
      } else if ([2,3,4].includes(count % 10) && ![12,13,14].includes(count % 100)) {
        fileWord = 'файла';
        chooseWord = 'о';
      } else {
        fileWord = 'файлов';
        chooseWord = 'о';
      }
      
      return `Выбран${chooseWord} ${count} ${fileWord} (${this.formatFileSize(totalSize)})`;
    }
  
    // Отрисовка списка файлов с предварительным просмотром
    async renderFileList() {
      this.fileList.innerHTML = '';
      this.fileList.classList.remove('empty');
      
      for (let i = 0; i < this.files.length; i++) {
        const file = this.files[i];
        const fileItem = document.createElement('li');
        fileItem.className = 'file-item';
        
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        
        const fileName = document.createElement('span');
        fileName.className = 'file-name';
        fileName.textContent = file.name;
        
        const fileSize = document.createElement('span');
        fileSize.className = 'file-size';
        fileSize.textContent = this.formatFileSize(file.size);
        
        // Добавляем индикатор загрузки для предварительного просмотра
        const previewIndicator = document.createElement('div');
        previewIndicator.className = 'preview-indicator';
        previewIndicator.innerHTML = '<span class="loading-spinner"></span> Анализ...';
        
        fileInfo.appendChild(fileName);
        fileInfo.appendChild(fileSize);
        fileInfo.appendChild(previewIndicator);
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-file';
        removeBtn.textContent = '×';
        removeBtn.title = 'Удалить файл';
        removeBtn.addEventListener('click', (e) => {
          e.preventDefault();
          this.removeFile(i);
        });
        
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(removeBtn);
        this.fileList.appendChild(fileItem);
        
        // Асинхронно загружаем предварительный просмотр
        try {
          const preview = await this.previewFile(file);
          if (preview) {
            previewIndicator.innerHTML = `
              <span class="file-stats">
                ${preview.lineCount} строк, ${preview.wordCount} слов
              </span>
            `;
            previewIndicator.className = 'preview-indicator loaded';
            
            // Добавляем тултип с предварительным просмотром
            fileItem.title = `Предварительный просмотр:\n${preview.preview}`;
          }
        } catch (error) {
          previewIndicator.innerHTML = '<span class="error">Ошибка чтения</span>';
          previewIndicator.className = 'preview-indicator error';
        }
      }
    }
  
    // Удаление файла из списка
    removeFile(index) {
      this.files.splice(index, 1);
      this.updateFileDisplay();
      hideDownloadButton();
    }

    // Очистка всех файлов
    clearFiles() {
      this.files = [];
      this.fileInput.value = '';
      this.updateFileDisplay();
      hideDownloadButton();
    }
  
    // Форматирование размера файла
    formatFileSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
  
    // Чтение содержимого файла с улучшенной обработкой ошибок
    readFileContent(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = e => {
          try {
            const content = e.target.result;
            // Проверяем кодировку и содержимое
            if (content.length === 0) {
              reject(new Error('Файл пустой'));
              return;
            }
            
            // Проверяем на наличие нечитаемых символов
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
        
        // Устанавливаем таймаут для больших файлов
        const timeout = setTimeout(() => {
          reader.abort();
          reject(new Error('Превышено время чтения файла'));
        }, 30000); // 30 секунд
        
        reader.onloadend = () => clearTimeout(timeout);
        
        reader.readAsText(file, 'UTF-8');
      });
    }
  
    // Получение текущего списка файлов (для других модулей)
    getFiles() {
      return this.files;
    }

    // Получение статистики файлов
    getFilesStats() {
      const totalSize = this.files.reduce((sum, file) => sum + file.size, 0);
      const avgSize = this.files.length > 0 ? totalSize / this.files.length : 0;
      
      return {
        count: this.files.length,
        totalSize,
        avgSize,
        largestFile: this.files.reduce((max, file) => file.size > max.size ? file : max, { size: 0 }),
        smallestFile: this.files.reduce((min, file) => file.size < min.size ? file : min, { size: Infinity })
      };
    }
  }
  
  // Создаем экземпляр и делаем его доступным глобально
  window.fileHandler = new FileHandler();