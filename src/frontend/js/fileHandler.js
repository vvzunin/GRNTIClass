class FileHandler {
    constructor() {
      this.files = [];
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
    }

    handleFileSelect() {
        return new Promise((resolve) => {
          this.files = Array.from(this.fileInput.files);
          this.updateFileDisplay();
          resolve();
        });
      }
  
    // Обработка перетаскивания файлов (над областью)
    handleDragOver(e) {
      e.preventDefault();
      this.fileUploadArea.classList.add('drag-over');
    }
  
    // Обработка когда файлы ушли за пределы области
    handleDragLeave() {
      this.fileUploadArea.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        this.fileUploadArea.classList.remove('drag-over');
        
        return new Promise((resolve) => {
          this.files = Array.from(e.dataTransfer.files).filter(file => 
            file.type === 'text/plain' || file.name.endsWith('.txt')
          );
          this.updateFileDisplay();
          resolve();
        });
      }
  
    // Обновление отображения информации о файлах
    updateFileDisplay() {
      document.getElementById('resultsSection').style.display = 'none';
      
      if (this.files.length === 0) {
        this.fileStatus.textContent = 'Файлы не выбраны';
        this.fileList.innerHTML = '';
        this.fileList.classList.add('empty');
        document.getElementById('classifyBtn').disabled = true;
      } else {
        this.fileStatus.textContent = this.getFilesCountText();
        this.renderFileList();
        document.getElementById('classifyBtn').disabled = false;
      }
    }
  
    // Формирование текста с количеством файлов
    getFilesCountText() {
      const count = this.files.length;
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
      
      return `Выбран${chooseWord} ${count} ${fileWord}`;
    }
  
    // Отрисовка списка файлов
    renderFileList() {
      this.fileList.innerHTML = '';
      this.fileList.classList.remove('empty');
      
      this.files.forEach((file, index) => {
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
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-file';
        removeBtn.textContent = '×';
        removeBtn.addEventListener('click', (e) => {
          e.preventDefault();
          this.removeFile(index);
        });
        
        fileInfo.appendChild(fileName);
        fileInfo.appendChild(fileSize);
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(removeBtn);
        this.fileList.appendChild(fileItem);
      });
    }
  
    // Удаление файла из списка
    removeFile(index) {
      this.files.splice(index, 1);
      this.updateFileDisplay();
    }
  
    // Форматирование размера файла
    formatFileSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
  
    // Чтение содержимого файла (если нужно в других модулях)
    readFileContent(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Ошибка чтения файла'));
        reader.readAsText(file);
      });
    }
  
    // Получение текущего списка файлов (для других модулей)
    getFiles() {
      return this.files;
    }
  }
  
  // Создаем экземпляр и делаем его доступным глобально
  window.fileHandler = new FileHandler();