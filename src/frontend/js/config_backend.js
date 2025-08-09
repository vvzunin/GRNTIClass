
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
})
.catch(error => {
    console.error('Error loading config:', error);
});
