class API {
    static async classify(files, params, progressCallback) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            
            files.forEach(file => formData.append('files', file));
            formData.append('level1', params.level1);
            formData.append('level2', params.level2);
            formData.append('level3', params.level3);
            formData.append('normalization', params.normalization);
            formData.append('decoding', params.decoding);
            formData.append('threshold', params.threshold);

            const results = [];
            let totalFiles = 0;
            let processingComplete = false;
            
            fetch('http://localhost:8000/classify', {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'text/event-stream'
                }
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                const processChunk = ({ done, value }) => {
                    if (done) {
                        if (!processingComplete) {
                            reject(new Error('Соединение закрыто до завершения обработки'));
                        }
                        return;
                    }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.trim() === '') continue;
                        
                        try {
                            console.log("RAW LINE FROM SSE:", line);

                            const data = JSON.parse(line);
                            
                            switch(data.type) {
                                case 'init':
                                    totalFiles = data.total_files;
                                    progressCallback(0, data.message, 0, totalFiles);
                                    break;
                                    
                                case 'progress':
                                    progressCallback(
                                        data.progress,
                                        data.message,
                                        data.completed + 1,
                                        data.total_files
                                        // totalFiles
                                    );
                                    break;
                                    
                                case 'result':
                                    results.push({
                                        filename: data.filename,
                                        rubrics: data.rubrics
                                    });
                                    break;
                                    
                                case 'complete':
                                    processingComplete = true;
                                    resolve(results);
                                    break;
                                    
                                case 'error':
                                    throw new Error(data.message);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                    
                    return reader.read().then(processChunk);
                };
                
                return reader.read().then(processChunk);
            }).catch(error => {
                reject(new Error(`Ошибка соединения: ${error.message}`));
            });
        });
    }
}
window.API = API;