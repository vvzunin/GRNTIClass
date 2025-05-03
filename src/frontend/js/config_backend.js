fetch('../static/config.json')
.then(response => response.json())
.then(config => {
    const apiUrl = `http://${config.api.host}:${config.api.port}/classify`;
    window.apiUrl = apiUrl;
    console.log("API URL:", apiUrl);
});