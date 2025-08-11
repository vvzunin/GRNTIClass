
// Глобальная переменная для отслеживания загрузки конфигурации
window.configLoaded = false;

fetch('config.json')
.then(response => {
    console.log('Config response status:', response.status);
    return response.json();
})
.then(config => {
    console.log('Loaded config:', config);
    const apiUrl = `http://${config.api.host}:${config.api.port}/classify`;
    window.apiUrl = apiUrl;
    console.log("API URL:", apiUrl);
    
    // Настройки API из конфигурации (если есть)
    if (config.api && config.api.settings) {
        if (config.api.settings.enableHealthCheck !== undefined) {
            API.config.enableHealthCheck = config.api.settings.enableHealthCheck;
        }
        if (config.api.settings.enableCompression !== undefined) {
            API.config.enableCompression = config.api.settings.enableCompression;
        }
        console.log('API settings loaded:', {
            enableHealthCheck: API.config.enableHealthCheck,
            enableCompression: API.config.enableCompression
        });
    }
    
    // Помечаем конфигурацию как загруженную
    window.configLoaded = true;
    
    // Запускаем проверку состояния сервера после загрузки конфигурации с небольшой задержкой
    if (window.initializeServerHealthCheck) {
        setTimeout(() => {
            window.initializeServerHealthCheck();
        }, 100); // Небольшая задержка для стабилизации
    }
})
.catch(error => {
    console.error('Error loading config:', error);
    // Даже при ошибке помечаем как загруженную, чтобы не блокировать
    window.configLoaded = true;
});
